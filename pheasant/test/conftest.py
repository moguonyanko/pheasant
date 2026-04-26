import logging

import pytest

_LOGGER_NAME = "PHEASANT_LOGGER"


@pytest.fixture(scope="session", autouse=True)
def setup_before_all_tests():
    # logger = logging.getLogger(_LOGGER_NAME)
    # # このロガー自身のレベルを DEBUG に設定
    # logger.setLevel(logging.DEBUG)

    # 事前処理
    logging.basicConfig(
        # ① ログレベルをDEBUGに設定: これでDEBUGメッセージも出力されます。
        level=logging.DEBUG,
        # ② ログのフォーマットを設定: 日時、レベル名、ロガー名、メッセージを含めます。
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    yield

    # 事後処理


@pytest.fixture(scope="function")
def pheasant_logger():
    return logging.getLogger(_LOGGER_NAME)
