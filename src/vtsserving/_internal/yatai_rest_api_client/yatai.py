import io
import logging
from typing import Optional
from urllib.parse import urljoin

import requests

from .schemas import UserSchema
from .schemas import VtsSchema
from .schemas import ModelSchema
from .schemas import schema_to_json
from .schemas import schema_from_json
from .schemas import CreateVtsSchema
from .schemas import CreateModelSchema
from .schemas import UpdateVtsSchema
from .schemas import OrganizationSchema
from .schemas import VtsRepositorySchema
from .schemas import ModelRepositorySchema
from .schemas import FinishUploadVtsSchema
from .schemas import FinishUploadModelSchema
from .schemas import CreateVtsRepositorySchema
from .schemas import CreateModelRepositorySchema
from .schemas import CompleteMultipartUploadSchema
from .schemas import PreSignMultipartUploadUrlSchema
from ...exceptions import YataiRESTApiClientError
from ..configuration import VTSSERVING_VERSION

logger = logging.getLogger(__name__)


class YataiRESTApiClient:
    def __init__(self, endpoint: str, api_token: str) -> None:
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-YATAI-API-TOKEN": api_token,
                "Content-Type": "application/json",
                "X-Vtsml-Version": VTSSERVING_VERSION,
            }
        )

    def _is_not_found(self, resp: requests.Response) -> bool:
        # Forgive me, I don't know how to map the error returned by gorm to juju/errors
        return resp.status_code == 400 and "record not found" in resp.text

    def _check_resp(self, resp: requests.Response) -> None:
        if resp.status_code != 200:
            raise YataiRESTApiClientError(
                f"request failed with status code {resp.status_code}: {resp.text}"
            )

    def get_current_user(self) -> Optional[UserSchema]:
        url = urljoin(self.endpoint, "/api/v1/auth/current")
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, UserSchema)

    def get_current_organization(self) -> Optional[OrganizationSchema]:
        url = urljoin(self.endpoint, "/api/v1/current_org")
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, OrganizationSchema)

    def get_vts_repository(
        self, vts_repository_name: str
    ) -> Optional[VtsRepositorySchema]:
        url = urljoin(
            self.endpoint, f"/api/v1/vts_repositories/{vts_repository_name}"
        )
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsRepositorySchema)

    def create_vts_repository(
        self, req: CreateVtsRepositorySchema
    ) -> VtsRepositorySchema:
        url = urljoin(self.endpoint, "/api/v1/vts_repositories")
        resp = self.session.post(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsRepositorySchema)

    def get_vts(
        self, vts_repository_name: str, version: str
    ) -> Optional[VtsSchema]:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}",
        )
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsSchema)

    def create_vts(
        self, vts_repository_name: str, req: CreateVtsSchema
    ) -> VtsSchema:
        url = urljoin(
            self.endpoint, f"/api/v1/vts_repositories/{vts_repository_name}/vtss"
        )
        resp = self.session.post(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsSchema)

    def update_vts(
        self, vts_repository_name: str, version: str, req: UpdateVtsSchema
    ) -> VtsSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}",
        )
        resp = self.session.patch(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsSchema)

    def presign_vts_upload_url(
        self, vts_repository_name: str, version: str
    ) -> VtsSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}/presign_upload_url",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsSchema)

    def presign_vts_download_url(
        self, vts_repository_name: str, version: str
    ) -> VtsSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}/presign_download_url",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsSchema)

    def start_vts_multipart_upload(
        self, vts_repository_name: str, version: str
    ) -> VtsSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}/start_multipart_upload",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsSchema)

    def presign_vts_multipart_upload_url(
        self,
        vts_repository_name: str,
        version: str,
        req: PreSignMultipartUploadUrlSchema,
    ) -> VtsSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}/presign_multipart_upload_url",
        )
        resp = self.session.patch(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsSchema)

    def complete_vts_multipart_upload(
        self,
        vts_repository_name: str,
        version: str,
        req: CompleteMultipartUploadSchema,
    ) -> VtsSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}/complete_multipart_upload",
        )
        resp = self.session.patch(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsSchema)

    def start_upload_vts(
        self, vts_repository_name: str, version: str
    ) -> VtsSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}/start_upload",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsSchema)

    def finish_upload_vts(
        self, vts_repository_name: str, version: str, req: FinishUploadVtsSchema
    ) -> VtsSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}/finish_upload",
        )
        resp = self.session.patch(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, VtsSchema)

    def upload_vts(
        self, vts_repository_name: str, version: str, data: io.BytesIO
    ) -> None:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}/upload",
        )
        resp = self.session.put(
            url,
            data=data,
            headers=dict(
                self.session.headers, **{"Content-Type": "application/octet-stream"}
            ),
        )
        self._check_resp(resp)
        return None

    def download_vts(
        self, vts_repository_name: str, version: str
    ) -> requests.Response:
        url = urljoin(
            self.endpoint,
            f"/api/v1/vts_repositories/{vts_repository_name}/vtss/{version}/download",
        )
        resp = self.session.get(url, stream=True)
        self._check_resp(resp)
        return resp

    def get_model_repository(
        self, model_repository_name: str
    ) -> Optional[ModelRepositorySchema]:
        url = urljoin(
            self.endpoint, f"/api/v1/model_repositories/{model_repository_name}"
        )
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelRepositorySchema)

    def create_model_repository(
        self, req: CreateModelRepositorySchema
    ) -> ModelRepositorySchema:
        url = urljoin(self.endpoint, "/api/v1/model_repositories")
        resp = self.session.post(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelRepositorySchema)

    def get_model(
        self, model_repository_name: str, version: str
    ) -> Optional[ModelSchema]:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}",
        )
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def create_model(
        self, model_repository_name: str, req: CreateModelSchema
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint, f"/api/v1/model_repositories/{model_repository_name}/models"
        )
        resp = self.session.post(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def presign_model_upload_url(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/presign_upload_url",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def presign_model_download_url(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/presign_download_url",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def start_model_multipart_upload(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/start_multipart_upload",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def presign_model_multipart_upload_url(
        self,
        model_repository_name: str,
        version: str,
        req: PreSignMultipartUploadUrlSchema,
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/presign_multipart_upload_url",
        )
        resp = self.session.patch(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def complete_model_multipart_upload(
        self,
        model_repository_name: str,
        version: str,
        req: CompleteMultipartUploadSchema,
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/complete_multipart_upload",
        )
        resp = self.session.patch(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def start_upload_model(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/start_upload",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def finish_upload_model(
        self, model_repository_name: str, version: str, req: FinishUploadModelSchema
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/finish_upload",
        )
        resp = self.session.patch(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def upload_model(
        self, model_repository_name: str, version: str, data: io.BytesIO
    ) -> None:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/upload",
        )
        resp = self.session.put(
            url,
            data=data,
            headers=dict(
                self.session.headers, **{"Content-Type": "application/octet-stream"}
            ),
        )
        self._check_resp(resp)
        return None

    def download_model(
        self, model_repository_name: str, version: str
    ) -> requests.Response:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/download",
        )
        resp = self.session.get(url, stream=True)
        self._check_resp(resp)
        return resp
