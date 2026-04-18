document.addEventListener('DOMContentLoaded', () => {
    const grid = document.getElementById('wardrobe-grid');
    const emptyState = document.getElementById('empty-state');

    function renderWardrobe() {
        chrome.storage.local.get({ qlothi_wishlist: [] }, (res) => {
            const items = res.qlothi_wishlist;
            
            if (!items || items.length === 0) {
                grid.style.display = 'none';
                emptyState.style.display = 'flex';
                return;
            }

            grid.style.display = 'flex';
            emptyState.style.display = 'none';
            grid.innerHTML = ''; // clear

            items.forEach((item, index) => {
                const card = document.createElement('a');
                card.href = item.link || '#';
                card.target = '_blank';
                card.className = 'wardrobe-item';
                
                const initials = (item.store || '?')[0].toUpperCase();

                card.innerHTML = `
                    <div class="w-img-box">
                        <img src="${item.image}" alt="Saved item" onerror="this.style.display='none';">
                    </div>
                    <div class="w-info">
                        <div class="w-brand">${item.store || 'Store'}</div>
                        <div class="w-name">${item.name || 'Product'}</div>
                        <div class="w-price">${item.price && item.price !== '—' ? item.price : 'Check Price'}</div>
                    </div>
                    <button class="remove-btn" title="Remove from wardrobe">✕</button>
                `;

                // Handle removal
                const rmBtn = card.querySelector('.remove-btn');
                rmBtn.addEventListener('click', (e) => {
                    e.preventDefault(); // prevent opening the link
                    e.stopPropagation();
                    
                    const newList = items.filter(i => i.link !== item.link);
                    chrome.storage.local.set({ qlothi_wishlist: newList }, () => {
                        renderWardrobe(); // Re-render
                    });
                });

                grid.appendChild(card);
            });
        });
    }

    renderWardrobe();
});
