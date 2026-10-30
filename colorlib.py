"""Differentiable colour-science utilities used throughout ChromaCert.

All functions accept tensors whose last dimension holds three channels.
sRGB values are in [0, 1]. Every function is written with torch operations
so that gradients flow through harmony and accessibility terms.
"""
import torch

# --- sRGB transfer ---------------------------------------------------------

def srgb_to_linear(c):
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).clamp_min(1e-12) ** 2.4)


def linear_to_srgb(c):
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


# --- OKLab (Ottosson, 2020) -----------------------------------------------
_M1 = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = torch.tensor([[0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660]])
_M2I = torch.tensor([[1.0, 0.3963377774, 0.2158037573],
                     [1.0, -0.1055613458, -0.0638541728],
                     [1.0, -0.0894841775, -1.2914855480]])
_M1I = torch.tensor([[4.0767416621, -3.3077115913, 0.2309699292],
                     [-1.2684380046, 2.6097574011, -0.3413193965],
                     [-0.0041960863, -0.7034186147, 1.7076147010]])


def _cbrt(x):
    return torch.sign(x) * x.abs().clamp_min(1e-12) ** (1.0 / 3.0)


def srgb_to_oklab(rgb):
    lin = srgb_to_linear(rgb)
    lms = lin @ _M1.T.to(rgb)
    return _cbrt(lms) @ _M2.T.to(rgb)


def oklab_to_srgb(lab, clip=True):
    lms = (lab @ _M2I.T.to(lab)) ** 3
    lin = lms @ _M1I.T.to(lab)
    out = linear_to_srgb(lin.clamp(0, 1)) if clip else lin
    return out


def oklab_in_gamut(lab, tol=1e-4):
    lms = (lab @ _M2I.T.to(lab)) ** 3
    lin = lms @ _M1I.T.to(lab)
    return ((lin >= -tol) & (lin <= 1 + tol)).all(-1)


def oklab_to_oklch(lab):
    L, a, b = lab.unbind(-1)
    C = torch.sqrt(a * a + b * b + 1e-12)
    h = torch.atan2(b, a)
    return torch.stack([L, C, h], -1)


def oklch_to_oklab(lch):
    L, C, h = lch.unbind(-1)
    return torch.stack([L, C * torch.cos(h), C * torch.sin(h)], -1)


# --- CIELAB (D65) and CIEDE2000 -------------------------------------------
_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def srgb_to_cielab(rgb):
    xyz = srgb_to_linear(rgb) @ _RGB2XYZ.T.to(rgb)
    t = xyz / _WHITE.to(rgb)
    d = 6.0 / 29.0
    f = torch.where(t > d ** 3, t.clamp_min(1e-12) ** (1 / 3), t / (3 * d * d) + 4.0 / 29.0)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], -1)


def ciede2000(lab1, lab2):
    """CIEDE2000 colour difference following Sharma, Wu and Dalal (2005)."""
    L1, a1, b1 = lab1.unbind(-1)
    L2, a2, b2 = lab2.unbind(-1)
    C1 = torch.sqrt(a1 ** 2 + b1 ** 2 + 1e-12)
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2 + 1e-12)
    Cb = 0.5 * (C1 + C2)
    G = 0.5 * (1 - torch.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = torch.sqrt(a1p ** 2 + b1 ** 2 + 1e-12)
    C2p = torch.sqrt(a2p ** 2 + b2 ** 2 + 1e-12)
    h1p = torch.remainder(torch.rad2deg(torch.atan2(b1, a1p)), 360.0)
    h2p = torch.remainder(torch.rad2deg(torch.atan2(b2, a2p)), 360.0)
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = torch.where(dh > 180, dh - 360, torch.where(dh < -180, dh + 360, dh))
    dh = torch.where(C1p * C2p < 1e-9, torch.zeros_like(dh), dh)
    dHp = 2 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(dh) / 2)
    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    hs = h1p + h2p
    hbp = torch.where((h1p - h2p).abs() > 180,
                      torch.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2), hs / 2)
    hbp = torch.where(C1p * C2p < 1e-9, hs, hbp)
    T = (1 - 0.17 * torch.cos(torch.deg2rad(hbp - 30)) + 0.24 * torch.cos(torch.deg2rad(2 * hbp))
         + 0.32 * torch.cos(torch.deg2rad(3 * hbp + 6)) - 0.20 * torch.cos(torch.deg2rad(4 * hbp - 63)))
    dtheta = 30 * torch.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * torch.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / torch.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -torch.sin(torch.deg2rad(2 * dtheta)) * Rc
    return torch.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                      + Rt * (dCp / Sc) * (dHp / Sh) + 1e-12)


# --- Contrast: WCAG 2.2 and APCA 0.0.98G-4g --------------------------------

def rel_luminance(rgb):
    lin = srgb_to_linear(rgb)
    return lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722


def wcag_ratio(fg, bg):
    l1, l2 = rel_luminance(fg), rel_luminance(bg)
    hi, lo = torch.maximum(l1, l2), torch.minimum(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def apca_lc(txt, bg):
    """Absolute APCA lightness contrast |Lc| (text on background)."""
    co = torch.tensor([0.2126729, 0.7151522, 0.0721750]).to(txt)

    def y(c):
        yy = (c.clamp(0, 1) ** 2.4 * co).sum(-1)
        return torch.where(yy < 0.022, yy + (0.022 - yy).clamp_min(0) ** 1.414, yy)

    yt, yb = y(txt), y(bg)
    normal = (yb.clamp_min(1e-9) ** 0.56 - yt.clamp_min(1e-9) ** 0.57) * 1.14
    rev = (yb.clamp_min(1e-9) ** 0.65 - yt.clamp_min(1e-9) ** 0.62) * 1.14
    sapc = torch.where(yb > yt, normal, rev)
    out = torch.where(sapc.abs() < 0.1, torch.zeros_like(sapc),
                      torch.where(sapc > 0, sapc - 0.027, sapc + 0.027))
    out = torch.where((yb - yt).abs() < 0.0005, torch.zeros_like(out), out)
    return out.abs() * 100.0


# --- Colour-vision-deficiency simulation (Machado et al., 2009, severity 1) -
CVD = {
    'normal': torch.eye(3),
    'protan': torch.tensor([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    'deutan': torch.tensor([[0.367322, 0.860646, -0.227968],
                            [0.280085, 0.672501, 0.047413],
                            [-0.011820, 0.042940, 0.968881]]),
    'tritan': torch.tensor([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}


def simulate_cvd(rgb, kind):
    if kind == 'normal':
        return rgb
    lin = srgb_to_linear(rgb) @ CVD[kind].T.to(rgb)
    return linear_to_srgb(lin.clamp(0, 1))


def hex_to_rgb(h):
    h = h.strip().lstrip('#')
    if len(h) == 3:
        h = ''.join(ch * 2 for ch in h)
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
