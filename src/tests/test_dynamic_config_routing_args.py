import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_router.dynamic_config import DynamicConfigWatcher, DynamicRouterConfig
from vllm_router.routers import routing_logic as routing_logic_module
from vllm_router.routers.routing_logic import cleanup_routing_logic


@pytest.fixture(autouse=True)
def clean_router_singletons():
    cleanup_routing_logic()
    yield
    cleanup_routing_logic()


@pytest.fixture
def controller_manager(monkeypatch):
    """lmcache is optional, so stub the module the routers build their manager from."""
    module = SimpleNamespace(LMCacheControllerManager=MagicMock())
    module.LMCacheControllerManager.return_value.start_all = AsyncMock()
    monkeypatch.setattr(
        routing_logic_module, "controller_manager", module, raising=False
    )
    return module.LMCacheControllerManager


def startup_config(routing_logic):
    return DynamicRouterConfig(
        service_discovery="static",
        routing_logic=routing_logic,
        kv_aware_threshold=500,
        loadaware_beta=0.5,
        prefix_min_match_length=64,
        lmcache_controller_port=9100,
        lmcache_controller_reply_port=9101,
        lmcache_controller_heartbeat_port=9102,
        lmcache_health_check_interval=7,
        lmcache_worker_timeout=40,
    )


def reload(tmp_path, routing_logic, **file_fields):
    path = tmp_path / "dynamic_config.json"
    path.write_text(
        json.dumps(
            {"service_discovery": "static", "routing_logic": routing_logic}
            | file_fields
        )
    )
    watcher = SimpleNamespace(
        app=MagicMock(),
        config_path=str(path),
        config_file_type="JSON",
        init_config=startup_config(routing_logic),
    )
    config = DynamicConfigWatcher.load_config(watcher)
    DynamicConfigWatcher.reconfigure_routing_logic(watcher, config)
    return watcher.app.state.router


@pytest.mark.parametrize("routing_logic", ["kvaware", "loadaware"])
def test_reload_keeps_startup_kv_args(tmp_path, controller_manager, routing_logic):
    router = reload(tmp_path, routing_logic)

    assert router.threshold == 500
    controller_manager.assert_called_once_with(
        {
            "pull": "0.0.0.0:9100",
            "reply": "0.0.0.0:9101",
            "heartbeat": "0.0.0.0:9102",
        },
        health_check_interval=7,
        lmcache_worker_timeout=40,
    )


def test_reload_keeps_startup_loadaware_beta(tmp_path, controller_manager):
    assert reload(tmp_path, "loadaware").beta == 0.5


def test_reload_keeps_startup_prefix_min_match_length(tmp_path):
    assert reload(tmp_path, "prefixaware").prefix_min_match_length == 64


def test_reload_takes_routing_args_from_the_file(tmp_path, controller_manager):
    router = reload(
        tmp_path, "kvaware", kv_aware_threshold=300, lmcache_controller_port=9200
    )

    assert router.threshold == 300
    assert controller_manager.call_args.args[0]["pull"] == "0.0.0.0:9200"
