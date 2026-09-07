# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core import security
from app.models.user import User
from app.services.egress_guard import guarded_httpx_client, validate_outbound_url
from shared.logger import setup_logger
from shared.utils.crypto import decrypt_sensitive_data, is_data_encrypted

logger = setup_logger("dify_api")

router = APIRouter()


class DifyAppInfoRequest(BaseModel):
    """Request to get Dify app info"""

    api_key: str
    base_url: str = "https://api.dify.ai"


def _fetch_dify_api(api_url: str, api_key: str) -> Dict[str, Any]:
    """Fetch a Dify API endpoint and return the whitelisted response fields.

    The upstream URL must pass the outbound request policy, and only the
    fields the product consumes are returned so an arbitrary upstream cannot
    read arbitrary content into the caller's session. Upstream error bodies
    are never reflected to the caller.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info(f"Fetching Dify app info from: {api_url}")

    try:
        validate_outbound_url(api_url)
        with guarded_httpx_client(timeout=10.0) as client:
            response = client.get(api_url, headers=headers)
    except HTTPException:
        # Policy rejections propagate with their specific status and detail.
        raise
    except Exception as e:
        error_msg = f"Failed to connect to Dify API: {type(e).__name__}"
        logger.error(error_msg)
        raise HTTPException(status_code=502, detail="Failed to connect to Dify API")

    if response.status_code != 200:
        logger.error(f"Dify API returned HTTP {response.status_code} for {api_url}")
        raise HTTPException(
            status_code=502,
            detail=f"Dify API returned an error (HTTP {response.status_code})",
        )

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502, detail="Dify API returned an invalid response"
        )
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502, detail="Dify API returned an invalid response"
        )
    return data


@router.post("/app/info")
def get_dify_app_info(
    request: DifyAppInfoRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(security.get_current_user),
) -> Dict[str, Any]:
    """
    Get Dify application information using API key

    Uses Dify's /v1/info endpoint to retrieve basic app information.
    This can be used to validate the API key and get app details.

    Args:
        request: Contains api_key and base_url

    Returns:
        Whitelisted app information (name, description, mode, icon fields).
    """

    try:
        # Decrypt API key if it's encrypted
        api_key = request.api_key
        if api_key and is_data_encrypted(api_key):
            api_key = decrypt_sensitive_data(api_key) or api_key
            logger.info("Decrypted API key for Dify app info request")

        api_url = f"{request.base_url.rstrip('/')}/v1/info"
        data = _fetch_dify_api(api_url, api_key)

        # Only return the fields the product consumes; the upstream response
        # must never be relayed verbatim to the caller.
        app_info: Dict[str, Any] = {}
        for field in ("name", "description", "mode", "icon", "icon_background"):
            if field in data:
                app_info[field] = data[field]

        logger.info(
            f"Successfully fetched Dify app info: {app_info.get('name', 'Unknown')}"
        )
        return app_info

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.post("/app/parameters")
def get_dify_app_parameters(
    request: DifyAppInfoRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(security.get_current_user),
) -> Dict[str, Any]:
    """
    Get parameters schema for a Dify application

    Uses Dify's /v1/parameters endpoint to retrieve app input parameters schema.

    Args:
        request: Contains api_key and base_url

    Returns:
        Whitelisted parameters schema (user_input_form, system_parameters).
    """

    try:
        # Decrypt API key if it's encrypted
        api_key = request.api_key
        if api_key and is_data_encrypted(api_key):
            api_key = decrypt_sensitive_data(api_key) or api_key
            logger.info("Decrypted API key for Dify app parameters request")

        api_url = f"{request.base_url.rstrip('/')}/v1/parameters"
        data = _fetch_dify_api(api_url, api_key)

        app_parameters: Dict[str, Any] = {}
        for field in ("user_input_form", "system_parameters"):
            if field in data:
                app_parameters[field] = data[field]

        logger.info("Successfully fetched Dify app parameters")
        return app_parameters

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to fetch app parameters: {type(e).__name__}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail="Failed to fetch app parameters")
