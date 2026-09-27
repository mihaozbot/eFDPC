import os
import torch

_RING_BITS = 59
_M = 1 << _RING_BITS
_SCALE = 1 << 28
_HALF = _M >> 1
_NONCE_BITS = 60

class ShufflerPrivacy:
    def __init__(self, parent):
        self.parent = parent
        self._send_ctr = 0

    @staticmethod
    def _rand_bits(shape, bits, device):
        out = torch.zeros(shape, dtype=torch.int64, device=device)
        got = 0
        while got < bits:
            take = min(30, bits - got)
            chunk = torch.randint(0, 1 << take, shape, dtype=torch.int64, device=device)
            out = (out << take) | chunk
            got += take
        return out

    @staticmethod
    def _encode(x):
        v = torch.round(x.to(torch.float64) * _SCALE).to(torch.int64)
        return torch.remainder(v, _M)

    @staticmethod
    def _decode(v):
        v = torch.remainder(v, _M)
        v = torch.where(v >= _HALF, v - _M, v)
        return (v.to(torch.float64) / _SCALE).to(torch.float32)

    def _split(self, v_mod, m):
        shares, acc = [], torch.zeros_like(v_mod)
        for _ in range(m - 1):
            r = self._rand_bits(v_mod.shape, _RING_BITS, v_mod.device)
            shares.append(r)
            acc = torch.remainder(acc + r, _M)
        shares.append(torch.remainder(v_mod - acc, _M))
        return shares

    @torch.no_grad()
    def _ensure_key_buffers(self, model):
        for nm in ("_sh_hi", "_sh_lo", "_sh_qmu", "_sh_qn"):
            if not hasattr(model, nm):
                setattr(model, nm, None)

    def _ensure_S_inv(self, model):
        if hasattr(model, "S_inv") and (model.S_inv is not None):
            return
        if hasattr(model, "S"):
            model.S_inv = torch.empty_like(model.S)

    @torch.no_grad()
    def client_make_shuffler_payloads(self, *, num_shufflers, federated_model_params,
                                      round_id=0, client_id=0, decoy_ratio=0.0, q=0,
                                      sigma_share=1.0):
        cm = self.parent
        C = int(cm.c)
        dev = cm.mu.device
        m = int(max(1, num_shufflers))
        ModelCls = type(cm)
        payloads = [ModelCls(**federated_model_params) for _ in range(m)]
        for pm in payloads:
            if hasattr(pm, "to"):
                try:
                    pm.to(dev)
                except Exception:
                    pass
            pm.c = 0
            self._ensure_key_buffers(pm)
        if C <= 0:
            return payloads

        qmu = self._encode(cm.mu[:C].to(dev))
        qn = self._encode(cm.n[:C].to(dev))

        hi = self._rand_bits((C,), _NONCE_BITS, dev)
        lo = self._rand_bits((C,), _NONCE_BITS, dev)

        mu_sh = [qmu] if m == 1 else self._split(qmu, m)
        n_sh = [qn] if m == 1 else self._split(qn, m)

        for i, pm in enumerate(payloads):
            pm._sh_qmu = mu_sh[i].clone()
            pm._sh_qn = n_sh[i].clone()
            pm._sh_hi = hi.clone()
            pm._sh_lo = lo.clone()
            pm.c = C
        return payloads

    @torch.no_grad()
    def shuffler_receive_payload(self, payload_model):
        dst, src = self.parent, payload_model
        Cs = int(src.c)
        if Cs <= 0:
            return
        self._ensure_key_buffers(dst)
        self._ensure_key_buffers(src)

        def app(a, b):
            return b.clone() if a is None else torch.cat([a, b], dim=0)

        dst._sh_qmu = app(dst._sh_qmu, src._sh_qmu)
        dst._sh_qn = app(dst._sh_qn, src._sh_qn)
        dst._sh_hi = app(dst._sh_hi, src._sh_hi)
        dst._sh_lo = app(dst._sh_lo, src._sh_lo)
        dst.c = int(dst.c) + Cs

    @torch.no_grad()
    def shuffle(self):
        sm = self.parent
        C = int(sm.c)
        if C <= 1 or getattr(sm, "_sh_hi", None) is None:
            return
        gen = torch.Generator(device=sm._sh_hi.device)
        gen.manual_seed(int.from_bytes(os.urandom(8), "little"))
        perm = torch.randperm(C, generator=gen, device=sm._sh_hi.device)
        sm._sh_qmu = sm._sh_qmu[perm]
        sm._sh_qn = sm._sh_qn[perm]
        sm._sh_hi = sm._sh_hi[perm]
        sm._sh_lo = sm._sh_lo[perm]

    @torch.no_grad()
    def aggregator_reconstruct_from_shufflers_inplace(self, *, shufflers, require_all=True,
                                                      clamp_S_min=1e-12, clamp_n_min=0.0):
        out = self.parent
        dev = out.mu.device
        self._ensure_key_buffers(out)
        self._ensure_S_inv(out)

        shufflers = [s for s in shufflers if int(s.c) > 0 and getattr(s, "_sh_hi", None) is not None]
        if not shufflers:
            out.c = 0
            return

        maps, sets = [], []
        for sm in shufflers:
            hi = sm._sh_hi.detach().cpu().tolist()
            lo = sm._sh_lo.detach().cpu().tolist()
            d = {(int(hi[i]), int(lo[i])): i for i in range(int(sm.c))}
            maps.append(d)
            sets.append(set(d.keys()))

        keep = set.intersection(*sets) if require_all else set.union(*sets)
        if not keep:
            out.c = 0
            return

        keys = list(keep)
        M = len(keys)

        dmu = int(out.mu.shape[1])
        smu = torch.zeros((M, dmu), dtype=torch.int64, device=dev)
        sn = torch.zeros((M,), dtype=torch.int64, device=dev)

        for si, sm in enumerate(shufflers):
            mp = maps[si]
            present = [(j, mp[k]) for j, k in enumerate(keys) if k in mp]
            if not present:
                continue
            jj = torch.tensor([p[0] for p in present], dtype=torch.long)
            ii = torch.tensor([p[1] for p in present], dtype=torch.long)
            smu[jj] = torch.remainder(smu[jj] + sm._sh_qmu[ii], _M)
            sn[jj] = torch.remainder(sn[jj] + sm._sh_qn[ii], _M)

        import numpy as np
        mu_dec = self._decode(smu)
        n_dec = torch.clamp(self._decode(sn), min=float(clamp_n_min))
        order = torch.as_tensor(np.lexsort(mu_dec.detach().cpu().numpy().T[::-1]), dtype=torch.long)
        mu_dec, n_dec = mu_dec[order], n_dec[order]

        out.overseer.ensure_capacity(M)
        self._ensure_S_inv(out)
        out.mu[:M].copy_(mu_dec.to(out.mu.dtype))
        if hasattr(out, "mu_og") and (out.mu_og is not None):
            out.mu_og[:M].copy_(out.mu[:M])
        out.n[:M].copy_(n_dec.to(out.n.dtype))
        if hasattr(out, "S_inv") and (out.S_inv is not None):
            out.S_inv[:M].copy_((1.0 / torch.clamp(out.S[:M], min=float(clamp_S_min))).to(out.S_inv.dtype))
        out.c = M
