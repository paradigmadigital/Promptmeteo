import re


# API Validators

def validate_version_rest(api_version):
    # Lógica de validación para el protocolo REST
    return not re.compile(r"\d{1}\.\d\.\d").fullmatch(api_version)


def version_validation(api_version, api_protocol):
    if api_protocol == "REST":
        return validate_version_rest(api_version)
    else:
        raise ValueError(
            f"Not available value for `api_protocol`."
        )
