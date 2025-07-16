import configparser
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Literal, Optional, Type, TypeVar, get_args, get_origin

T = TypeVar("T")


def parse_overrides(overrides: list[str]):
    override_dict = {}
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"Invalid KV override: {kv}")
        key, val = kv.split("=", 1)
        override_dict[key] = val
    return override_dict


def config_to_dataclass(
    config_path: str, overrides: list[str], dataclass_type: Type[T]
) -> T:
    """Converts a config file (ini, cfg) to an instance of a dataclass"""
    if not is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type.__name__} must be a dataclass type")

    if not config_path:
        raise ValueError("Must provide a config file!")

    config = configparser.ConfigParser()
    config.read(config_path)

    overrides_dict = parse_overrides(overrides)

    init_values = {}

    for field in fields(dataclass_type):
        field_type = field.type
        value: Optional[Any] = None

        is_literal = get_origin(field_type) is Literal
        literal_values = get_args(field_type) if is_literal else None

        if field.name in overrides_dict:
            # Use the override if available
            if field_type is bool:
                value = bool(overrides_dict[field.name])
            elif field_type is int:
                value = int(overrides_dict[field.name])
            elif field_type is float:
                value = float(overrides_dict[field.name])
            elif is_literal:
                value = overrides_dict[field.name]
                if value not in literal_values:  # type:ignore
                    raise ValueError(
                        f"Value '{value}' not in allowed literals {literal_values}"
                    )
            else:
                value = overrides_dict[field.name]
        else:
            # Retrieve the value from config using the appropriate getter
            try:
                if field_type is bool:
                    value = config.getboolean("config", field.name)
                elif field_type is int:
                    value = config.getint("config", field.name)
                elif field_type is float:
                    value = config.getfloat("config", field.name)
                elif is_literal:
                    # If Literal, retrieve as string and check validity
                    value = config.get("config", field.name)
                    if value not in literal_values:
                        raise ValueError(
                            f"Value '{value}' not in allowed literals {literal_values}"
                        )
                else:
                    value = config.get("config", field.name)
            except (configparser.NoSectionError, configparser.NoOptionError):
                # Use the default value if the key is missing
                value = field.default if field.default is not MISSING else None
            except ValueError as e:
                # Handle type conversion and Literal validation errors
                print(
                    f"Warning: {e}. Using default value for '{field.name}' in section '[config]'."
                )
                value = field.default if field.default is not MISSING else None

        init_values[field.name] = value

    return dataclass_type(**init_values)
