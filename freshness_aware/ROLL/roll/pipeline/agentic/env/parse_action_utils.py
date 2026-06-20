import re


def _lookup_action(action_content, action_lookup):
    if action_lookup is None or len(action_lookup) == 0:
        return None

    rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
    return rev_action_lookup.get(action_content.lower())


def default_parser_action_func(text, action_pattern, action_lookup, special_token_list):
    if special_token_list is not None:
        for special_token in special_token_list:
            text = text.replace(special_token, "").strip()

    action = None
    match = re.search(action_pattern, text, re.DOTALL)
    if not match:
        stripped_text = text.strip()
        if stripped_text and action_lookup is not None and len(action_lookup) > 0:
            action = _lookup_action(stripped_text, action_lookup)
            if action is not None:
                action_info = {
                    "action": action,
                    "action_content": stripped_text,
                    "think_content": ""
                }
                return action_info
        action_info = {
            "action": action,
            "action_content": "",
            "think_content": ""
        }
        return action_info
    try:
        if len(match.groups()) == 1:
            think_content, action_content = "", match.group(1).strip()
        else:
            think_content, action_content = match.group(1).strip(), match.group(2).strip()
        action_content = action_content.strip()
        think_content = think_content.strip()

        action = action_content
        if action_lookup is not None and len(action_lookup) > 0:
            action = _lookup_action(action_content, action_lookup)

        action_info = {
            "action": action,
            "action_content": action_content,
            "think_content": think_content,
        }
        return action_info
    except Exception as e:
        print(f"Error parsing action: {[text]}")
        print(f"Error parsing action: {e}")
        action_info = {
            "action": action,
            "action_content": "",
            "think_content": ""
        }
        return action_info
