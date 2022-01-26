from typing import Callable
import globus_sdk
from globus_sdk.authorizers.access_token import AccessTokenAuthorizer
from globus_sdk.authorizers.refresh_token import RefreshTokenAuthorizer
from globus_sdk.services.auth.client.native_client import NativeAppAuthClient

class DynamicStr(str):
    def __init__(self,fn: Callable[[], str]):
        self.fn = fn

    def __repr__(self) -> str:
        return self.fn()

def authorize(client_id):
    """
    Perform an initial (interactive) application authorization
    """

    client = globus_sdk.NativeAppAuthClient(client_id)
    client.oauth2_start_flow(refresh_tokens=True)

    authorize_url = client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login: {authorize_url}")
    auth_code = input("Auth code: ").strip()

    return client.oauth2_exchange_code_for_tokens(auth_code)

def transfer_client(client_id, refresh_token):
    client = NativeAppAuthClient(client_id)
    authorizer = RefreshTokenAuthorizer(refresh_token, client)
    return globus_sdk.TransferClient(authorizer=authorizer)
