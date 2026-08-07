# 业务模块包：保持轻量，避免在包初始化时级联导入全部子模块。
# 请从子模块直接导入，例如：from modules.watchdesk import render_watchdesk

__all__ = [
    "render_data_stats",
    "render_regression",
    "render_finance",
    "render_portfolio",
    "render_market_data",
    "render_ml_classify",
    "render_watchdesk",
]


def __getattr__(name: str):
    """按需懒加载，降低 Cloud 启动时的循环导入风险。"""
    mapping = {
        "render_data_stats": "modules.data_stats",
        "render_regression": "modules.regression",
        "render_finance": "modules.finance",
        "render_portfolio": "modules.portfolio",
        "render_market_data": "modules.market_data",
        "render_ml_classify": "modules.ml_classify",
        "render_watchdesk": "modules.watchdesk",
    }
    if name not in mapping:
        raise AttributeError(f"module 'modules' has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(mapping[name])
    return getattr(mod, name)
