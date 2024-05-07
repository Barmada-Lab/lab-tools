from cvat_sdk import Client, Config

from cytomancer.config import CytomancerConfig


def new_client_from_config(settings: CytomancerConfig):
    client = Client(url=settings.cvat_url, config=Config(verify_ssl=False))
    client.login((settings.cvat_username, settings.cvat_password))

    org_slug = settings.cvat_org
    client.organization_slug = org_slug
    return client
