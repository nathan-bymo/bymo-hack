import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useCatalogProduct, useUpdateCatalogImage, useUpdateCatalogProduct } from '../api/catalog'
import { AppShell } from '../components/app'
import { Breadcrumbs, ConfidenceMeter, Eyebrow, StatusTag, ValueTable } from '../components/catalog-ui'
import { ErrorState, useToast } from '../components/ui'
import { T, useI18n } from '../i18n'
import { FILTER_KEYS, filterLabel, percent, perspectiveLabel, ruleBlockers, viewOrderRule } from '../lib/catalog'
import { categoryLabel, plainLabel } from '../lib/labels'

export default function CatalogProductPage() {
  const { t, tn } = useI18n()
  const { productId } = useParams()
  const detail = useCatalogProduct(productId)
  const updateImage = useUpdateCatalogImage(productId)
  const updateProduct = useUpdateCatalogProduct(productId)
  const toast = useToast()
  const [form, setForm] = useState<Record<string, string>>({})
  const product = detail.data?.product
  const filters = detail.data?.filters

  useEffect(() => {
    if (!product) return
    setForm({
      title: product.title,
      brand: product.brand ?? '',
      category: product.category ?? '',
      subcategory: product.subcategory ?? '',
      product_type: product.product_type ?? '',
    })
  }, [product?.product_id])   // eslint-disable-line react-hooks/exhaustive-deps

  if (detail.isError) {
    return (
      <AppShell>
        <Breadcrumbs items={[{ label: t('Catalog'), to: '/catalog' }, { label: t('Product') }]} />
        <div className="mt-6"><ErrorState error={detail.error} /></div>
      </AppShell>
    )
  }
  if (detail.isLoading || !product || !filters) {
    return <AppShell><p className="mt-8 text-[15px] text-slate">{t('Loading the product…')}</p></AppShell>
  }

  const images = detail.data?.images ?? []
  const live = product.published_count > 0
  const saveProduct = () => {
    if (!form.title?.trim()) return
    updateProduct.mutate({
      title: form.title.trim(),
      brand: form.brand || null,
      category: form.category || null,
      subcategory: form.subcategory || null,
      product_type: form.product_type || null,
    }, { onSuccess: () => toast('ok', t('Product details saved.')) })
  }

  return (
    <AppShell>
      <Breadcrumbs items={[{ label: t('Catalog'), to: '/catalog' }, { label: product.title }]} />

      <div className="mt-4 flex flex-wrap items-end justify-between gap-4 border-b border-graphite pb-4">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <Eyebrow>{product.brand ?? t('No brand')}</Eyebrow>
            <StatusTag status={product.status} />
            <span className="text-[12px] text-slate">{product.product_id}</span>
          </div>
          <h1 className="mt-1.5 text-[36px] leading-[1.05]">{product.title}</h1>
          <p className="m-0 mt-2 text-[15px] text-slate">
            {[product.category ? categoryLabel(product.category) : null, product.subcategory, product.product_type].filter(Boolean).join(' · ') || t('Category not given')}
            {' · '}{t('{published} of {total} photos live', { published: product.published_count, total: product.image_count })}
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          {live && (
            <Link to={`/shop/${product.product_id}`} data-testid="catalog-view-shop"
              className="bg-graphite px-4 py-2 text-[14px] font-bold text-white no-underline hover:bg-signal">
              {t('View in the storefront →')}
            </Link>
          )}
          <Link to="/" className="border border-graphite px-4 py-2 text-[14px] font-bold no-underline hover:bg-canvas">
            {t('Add a photo')}
          </Link>
        </div>
      </div>

      <div className="mt-7 grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_340px]">
        {/* ---- foto del prodotto ---- */}
        <div>
          <div className="flex items-baseline justify-between border-b border-graphite pb-2">
            <h2 className="m-0 text-[20px] font-bold tracking-tight">{t('Photos and views')}</h2>
            <span className="text-[13px] text-slate">{tn(images.length, '{n} photo', '{n} photos')}</span>
          </div>
          <p className="m-0 mt-2 text-[13px] text-slate" data-testid="view-order-rule">{t('Order of the views:')} {viewOrderRule()}</p>

          {images.length === 0 ? (
            <div className="mt-4 border border-rule px-6 py-10 text-center" data-testid="catalog-product-empty">
              <p className="m-0 text-[18px] font-bold">{t('No photo yet.')}</p>
              <p className="m-0 mt-1 text-[14px] text-slate">{t('Upload a photo on the Classify page, classify it, then add it to this product.')}</p>
              <Link to="/" className="mt-3 inline-block border border-graphite px-4 py-2 text-[14px] font-bold no-underline hover:bg-canvas">{t('Add a photo')}</Link>
            </div>
          ) : (
            <div className="mt-4 space-y-3">
              {images.map((image) => {
                const blockers = image.catalog.published ? ruleBlockers(image.values, image.confidence, filters) : []
                return (
                  <article key={image.image_id} data-testid="catalog-image" data-image={image.image_id}
                    className={`border bg-white ${image.catalog.is_cover ? 'border-graphite' : 'border-rule'}`}>
                    <div className="flex flex-col gap-4 p-3 sm:flex-row">
                      <img src={image.file_url} alt={`${product.title}, ${perspectiveLabel(image.values.perspective)}`}
                        className="h-32 w-32 shrink-0 border border-rule bg-canvas object-contain p-1" />
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="truncate text-[12px] text-slate" title={image.original_filename}>{image.original_filename}</div>
                            <div className="text-[17px] font-bold leading-snug tracking-tight">{perspectiveLabel(image.values.perspective)}</div>
                            <div className="text-[13px] text-slate">
                              {image.values.subject ? plainLabel('subject', image.values.subject) : t('subject not read')}
                              {' · '}{image.values.background ? plainLabel('background', image.values.background) : t('background not read')}
                              {' · '}{image.values.individuals === 'none' ? t('no people') : image.values.individuals === 'individual_present' ? t('someone present') : t('people not read')}
                            </div>
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {image.catalog.is_cover && (
                              <span className="bg-graphite px-2 py-0.5 text-[11px] font-bold uppercase tracking-wide text-white">{t('Cover')}</span>
                            )}
                            <StatusTag status={image.catalog.published ? 'live' : 'draft'} />
                          </div>
                        </div>

                        {image.confidence !== null && <div className="mt-2"><ConfidenceMeter value={image.confidence} /></div>}

                        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-slate">
                          <span>{t('Values from:')} <b className="text-graphite">{image.values_source === 'review' ? t('human review') : image.values_source === 'model' ? t('the model') : t('no answer yet')}</b></span>
                          {image.model && <span>{image.model.model_id} · {image.model.mode === 'recorded' ? t('recorded') : t('live')}</span>}
                          {image.values.text_classification && image.values.text_classification !== 'none' && (
                            <span>{t('added text:')} {plainLabel('text_classification', image.values.text_classification)}</span>
                          )}
                        </div>

                        {blockers.length > 0 && (
                          <p className="m-0 mt-1.5 border-l-2 border-signal pl-2 text-[13px] font-bold text-signal" data-testid="image-hidden">
                            {t('Hidden by shop rules: {reasons}.', { reasons: blockers.join(', ') })}
                          </p>
                        )}

                        <div className="mt-2.5 flex flex-wrap items-center gap-3">
                          <label className="flex cursor-pointer items-center gap-2 text-[13px] font-bold">
                            <input type="checkbox" checked={image.catalog.published} disabled={updateImage.isPending}
                              data-testid="image-published"
                              onChange={(e) => updateImage.mutate({ imageId: image.image_id, published: e.target.checked })}
                              className="h-4 w-4 accent-signal" />
                            {t('Published')}
                          </label>
                          <button type="button" disabled={image.catalog.is_cover || updateImage.isPending}
                            data-testid="image-cover"
                            onClick={() => updateImage.mutate({ imageId: image.image_id, is_cover: true })}
                            className="border border-rule bg-white px-2.5 py-1 text-[12px] font-bold hover:border-graphite disabled:cursor-not-allowed disabled:opacity-40">
                            {image.catalog.is_cover ? t('Cover photo') : t('Make cover')}
                          </button>
                          <span className="text-[12px] text-slate">{t('position {n}', { n: image.catalog.position + 1 })}</span>
                        </div>
                      </div>
                    </div>
                  </article>
                )
              })}
            </div>
          )}
        </div>

        {/* ---- dettagli e anteprima ---- */}
        <div className="space-y-5 lg:sticky lg:top-6">
          <section className="border border-rule bg-white">
            <h2 className="m-0 border-b-2 border-graphite px-4 py-2.5 text-[17px] font-bold tracking-tight">{t('Product details')}</h2>
            <div className="space-y-2 px-4 py-3">
              {([['title', T('Title')], ['brand', T('Brand')], ['category', T('Category')], ['subcategory', T('Subcategory')], ['product_type', T('Product type')]] as const).map(([key, label]) => (
                <label key={key} className="block text-[12px] text-slate">{t(label)}
                  <input value={form[key] ?? ''} data-testid={`product-${key}`}
                    onChange={(e) => setForm({ ...form, [key]: e.target.value })}
                    className="mt-0.5 w-full border border-rule px-2.5 py-1.5 text-[14px] text-graphite focus:border-graphite" />
                </label>
              ))}
              <button type="button" onClick={saveProduct} disabled={updateProduct.isPending || !form.title?.trim()}
                data-testid="save-product"
                className="mt-1 w-full border border-graphite bg-white px-3 py-2 text-[14px] font-bold hover:bg-canvas disabled:opacity-50">
                {updateProduct.isPending ? t('Saving…') : t('Save details')}
              </button>
              <p className="m-0 text-[12px] text-slate" role="status">
                {updateProduct.isSuccess ? t('Saved.') : updateProduct.isError ? t('Not saved. Check the server.') : t('The storefront updates at once.')}
              </p>
            </div>
          </section>

          <section className="border border-rule bg-white" data-testid="catalog-preview">
            <h2 className="m-0 border-b-2 border-graphite px-4 py-2.5 text-[17px] font-bold tracking-tight">{t('Storefront preview')}</h2>
            <div className="px-4 py-3">
              {product.cover ? (
                <>
                  <div className="flex aspect-[4/3] items-center justify-center bg-canvas p-4">
                    <img src={product.cover.file_url} alt="" className="max-h-full max-w-full object-contain" />
                  </div>
                  <div className="mt-2 text-[11px] font-bold uppercase tracking-[0.12em] text-slate">{product.brand ?? t('No brand')}</div>
                  <div className="text-[15px] font-bold leading-snug">{product.title}</div>
                  <div className="mt-1 text-[12px] text-slate">{tn(product.published_count, '{n} live view', '{n} live views')}</div>
                  <Link to={`/shop/${product.product_id}`} className="mt-2 inline-block text-[13px] font-bold">{t('Open in the storefront →')}</Link>
                </>
              ) : (
                <p className="m-0 text-[13px] text-slate">{t('Publish a photo to see the card here.')}</p>
              )}
            </div>
          </section>

          <section className="border border-rule bg-white">
            <h2 className="m-0 border-b-2 border-graphite px-4 py-2.5 text-[17px] font-bold tracking-tight">{t('Shop rules')}</h2>
            <div className="px-4 py-3 text-[13px]">
              <ul className="m-0 list-none space-y-1.5 p-0">
                {FILTER_KEYS.map((key) => (
                  <li key={key} className="flex items-baseline justify-between gap-3">
                    <span className="text-slate">{filterLabel(key).title}</span>
                    <span className="font-bold">
                      {key === 'min_confidence' ? percent(filters.min_confidence) : filters[key] ? t('On') : t('Off')}
                    </span>
                  </li>
                ))}
              </ul>
              <Link to="/catalog" className="mt-3 inline-block text-[13px] font-bold">{t('Edit the rules →')}</Link>
            </div>
          </section>

          <section className="border border-rule bg-white">
            <h2 className="m-0 border-b-2 border-graphite px-4 py-2.5 text-[17px] font-bold tracking-tight">{t('Latest answer')}</h2>
            <div className="px-4 py-3">
              {images[0] ? <ValueTable values={images[0].values} confidence={images[0].confidence} compact /> : <p className="m-0 text-[13px] text-slate">{t('No answer yet.')}</p>}
              <p className="m-0 mt-2 text-[12px] leading-snug text-slate">{t('Values come from the latest model answer, unless a human review replaced them.')}</p>
            </div>
          </section>
        </div>
      </div>
    </AppShell>
  )
}
