from typing import Type, Union

from prl.environment.Wrappers.base import ActionSpace, ActionSpaceMinimal


def parse_cmd_action_to_action_cls(action: str) -> Union[
    ActionSpaceMinimal,  # Allow dichotomizers only for FOLD,CHECK,RAISE (Single bet size)
    Type[ActionSpaceMinimal],
    Type[ActionSpace]
]:
    if action == 'ActionSpace':
        return ActionSpace
    elif action == 'ActionSpaceMinimal':
        return ActionSpaceMinimal
    elif action.casefold().strip() == 'fold':
        return ActionSpaceMinimal.FOLD
    elif 'check' in action.casefold().strip():
        return ActionSpaceMinimal.CHECK_CALL
    elif action.casefold().strip() == 'raise':
        return ActionSpaceMinimal.RAISE
    else:
        raise NotImplementedError
