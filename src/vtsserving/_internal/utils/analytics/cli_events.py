import typing as t
from typing import TYPE_CHECKING

from .. import calc_dir_size
from .schemas import VtsBuildEvent

if TYPE_CHECKING:
    from ...vts.vts import Vts


def _cli_vtsserving_build_event(
    cmd_group: str,
    cmd_name: str,
    return_value: "t.Optional[Vts]",
) -> VtsBuildEvent:  # pragma: no cover
    if return_value is not None:
        vts = return_value
        return VtsBuildEvent(
            cmd_group=cmd_group,
            cmd_name=cmd_name,
            vts_creation_timestamp=vts.info.creation_time,
            vts_size_in_kb=calc_dir_size(vts.path_of("/")) / 1024,
            model_size_in_kb=calc_dir_size(vts.path_of("/models")) / 1024,
            num_of_models=len(vts.info.models),
            num_of_runners=len(vts.info.runners),
            model_types=[m.module for m in vts.info.models],
            runnable_types=[r.runnable_type for r in vts.info.runners],
        )
    else:
        return VtsBuildEvent(
            cmd_group=cmd_group,
            cmd_name=cmd_name,
        )


cli_events_map = {"cli": {"build": _cli_vtsserving_build_event}}
