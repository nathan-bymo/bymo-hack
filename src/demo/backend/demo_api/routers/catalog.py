"""API del catalogo e della vetrina (lato admin e lato cliente)."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from .. import catalog as C

router = APIRouter()


class ProductBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    brand: str | None = Field(default=None, max_length=120)
    category: str | None = Field(default=None, max_length=120)
    subcategory: str | None = Field(default=None, max_length=120)
    product_type: str | None = Field(default=None, max_length=120)


class ProductUpdateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=200)
    brand: str | None = Field(default=None, max_length=120)
    category: str | None = Field(default=None, max_length=120)
    subcategory: str | None = Field(default=None, max_length=120)
    product_type: str | None = Field(default=None, max_length=120)


class ImageUpdateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str | None = None
    published: bool | None = None
    position: int | None = Field(default=None, ge=0, le=1000)
    is_cover: bool | None = None
    note: str | None = Field(default=None, max_length=500)


class FiltersBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hide_non_photos: bool = True
    only_white_background: bool = False
    hide_people: bool = False
    hide_added_text: bool = False
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@router.get("/catalog")
def get_catalog(request: Request, q: str | None = Query(default=None, max_length=200),
                status: str | None = Query(default=None, pattern="^(live|draft|empty)$")):
    return C.list_products(request.app.state.db, q, status)


@router.post("/catalog/products", status_code=201)
def post_product(body: ProductBody, request: Request):
    detail, created = C.create_product(request.app.state.db, body.model_dump())
    return {"product": detail["product"], "images": detail["images"], "created": created}


@router.get("/catalog/products/{product_id}")
def get_product(product_id: str, request: Request):
    return C.product_detail(request.app.state.db, product_id)


@router.put("/catalog/products/{product_id}")
def put_product(product_id: str, body: ProductUpdateBody, request: Request):
    return C.update_product(request.app.state.db, product_id, body.model_dump(exclude_none=True))


@router.put("/catalog/images/{image_id}")
def put_image(image_id: str, body: ImageUpdateBody, request: Request):
    return C.update_image(request.app.state.db, image_id, body.model_dump(exclude_none=True))


@router.get("/storefront")
def get_storefront(request: Request, q: str | None = Query(default=None, max_length=200),
                   category: str | None = Query(default=None, max_length=120)):
    return C.storefront(request.app.state.db, q, category)


@router.get("/storefront/settings")
def get_storefront_settings(request: Request):
    return C.get_filters(request.app.state.db)


@router.put("/storefront/settings")
def put_storefront_settings(body: FiltersBody, request: Request):
    return C.set_filters(request.app.state.db, body.model_dump())


@router.get("/storefront/products/{product_id}")
def get_storefront_product(product_id: str, request: Request):
    return C.storefront_product(request.app.state.db, product_id)
