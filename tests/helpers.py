import io
from PIL import Image
from cme.atoms import Atom


def png_bytes(color=(120, 120, 120), size=(64, 96), patch=None, blank=False):
    """Synthetic frame. By default it carries a thin high-contrast stripe band so it is not a blank frame."""
    im = Image.new("RGB", size, color)
    if not blank:
        for x in range(0, size[0], 4):
            im.paste((20, 20, 20), (x, size[1] - 6, min(x + 2, size[0]), size[1]))
            im.paste((235, 235, 235), (min(x + 2, size[0]), size[1] - 6, min(x + 4, size[0]), size[1]))
    if patch:
        box, c = patch
        im.paste(c, box)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def A(i, text, prio="MUST", ch="image", shot=1, kind="scene"):
    return Atom(i, kind, text, prio, ch, shot)
