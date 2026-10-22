import io
from PIL import Image
from cme.atoms import Atom


def png_bytes(color=(120, 120, 120), size=(64, 96), patch=None):
    im = Image.new("RGB", size, color)
    if patch:
        box, c = patch
        im.paste(c, box)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def A(i, text, prio="MUST", ch="image", shot=1, kind="scene"):
    return Atom(i, kind, text, prio, ch, shot)
