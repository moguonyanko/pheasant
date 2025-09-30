"""
Imageを扱うパッケージの振る舞いをテストします。
"""

import io
from PIL import Image, ImageChops


def test_open_image_in_memory():
    """
    Imageをメモリ上に書き出し、それを再度読み込んで元のImageと同じであることを確認します。
    """
    target_img = Image.new("RGB", (100, 100), color="red")
    img_buffer = io.BytesIO()
    target_img.save(img_buffer, format="PNG")

    img_bytes = img_buffer.getvalue()

    assert len(img_bytes) > 0

    img_for_test = Image.open(io.BytesIO(img_bytes))

    # 差分画像を取得します。完全に同じなら真っ黒な画像になります。
    diff = ImageChops.difference(target_img, img_for_test)

    # getbbox() で差分画像の境界ボックスを取得して確認
    # getbbox() は、画像内の非ゼロピクセル（つまり、差分があるピクセル）を囲む
    # ボックス (left, upper, right, lower) を返します。
    # 差分が全くない（すべてゼロピクセル）場合、Noneを返します。

    assert diff.getbbox() is None
