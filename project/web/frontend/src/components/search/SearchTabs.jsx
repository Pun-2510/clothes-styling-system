import Icon from '../common/Icon'

const TABS = [
  { value: 'text', label: 'Tìm bằng mô tả', icon: 'search' },
  { value: 'image', label: 'Tìm bằng hình ảnh', icon: 'image' },
]

export default function SearchTabs({ activeTab, onChange }) {
  return (
    <div className="search-tabs" role="tablist" aria-label="Phương thức tìm kiếm">
      {TABS.map((tab) => (
        <button
          key={tab.value}
          type="button"
          role="tab"
          aria-selected={activeTab === tab.value}
          className={activeTab === tab.value ? 'active' : ''}
          onClick={() => onChange(tab.value)}
        >
          <Icon name={tab.icon} size={18} />
          {tab.label}
        </button>
      ))}
    </div>
  )
}
