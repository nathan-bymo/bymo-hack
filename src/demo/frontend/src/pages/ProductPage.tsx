import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { Session } from '../api/types'
import { ActionToolbar, CollapsibleSection, ErpShell, LabelValueGrid, SubLinks, TitleBar, ToolbarButton, ToolbarDropdown, WidgetCard } from '../components/shell'

export default function ProductPage() {
  const sessions = useQuery({ queryKey: ['sessions'], queryFn: () => api<{ sessions: Session[] }>('/sessions').then((r) => r.sessions) })
  const current = sessions.data?.[0]
  const right = (
    <>
      <WidgetCard title="Working Time" />
      <WidgetCard title="Notifications" />
      <WidgetCard title="User selection" />
      <WidgetCard title="Barcode" />
      <WidgetCard title="Sales Assistant" />
      <WidgetCard title="Packing info" />
    </>
  )
  return (
    <ErpShell rightColumn={right}>
      <TitleBar title="Products — Demo product" id="DEMO-0001" />
      <SubLinks links={[{ to: '/product/DEMO-0001', label: 'Search' }, { to: '/product/DEMO-0001', label: 'Advanced search' }]} />
      <ActionToolbar>
        <ToolbarDropdown label="Enrich Product" items={[
          { label: 'Image classification (AI)', to: '/image-classification', testId: 'menu-image-classification' },
          { label: 'Translate texts', disabled: true },
        ]} />
        <ToolbarDropdown label="Shops" items={[{ label: 'Publish', disabled: true }]} />
        <ToolbarDropdown label="Deactivate" items={[{ label: 'Deactivate product', disabled: true }]} />
        <ToolbarDropdown label="Methods" items={[{ label: 'Run method', disabled: true }]} />
        <ToolbarButton disabled title="Not active in this demo">Log</ToolbarButton>
      </ActionToolbar>
      <CollapsibleSection title="Product Overview">
        <LabelValueGrid rows={[
          { label: 'Name', value: 'Demo product' },
          { label: 'Id', value: 'DEMO-0001' },
          { label: 'Images', value: `${current?.image_count ?? 0} (demo session)`, info: 'Count of images uploaded to the demo session.' },
          { label: 'Image classification', value: 'Available under Enrich Product ▾' },
        ]} />
        <p className="mt-3 text-xs text-muted">
          Static demo shell: no connection to the real ERP. Everything here is placeholder text.
        </p>
      </CollapsibleSection>
      <CollapsibleSection title="Provider order backlog" inactive />
      <CollapsibleSection title="Chosen Sales Offer" inactive />
      <CollapsibleSection title="Retail information" inactive />
    </ErpShell>
  )
}
