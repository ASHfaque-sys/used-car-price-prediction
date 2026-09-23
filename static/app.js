document.addEventListener('DOMContentLoaded', () => {
  const form = document.querySelector('#prediction-form');
  const submitSingleBtn = document.querySelector('#submit-single-btn');
  const clearFormBtn = document.querySelector('#clear-form-btn');

  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  const idleState = document.querySelector('#result-idle');
  const outputState = document.querySelector('#result-output');
  const loadingState = document.querySelector('#loading');
  const errorState = document.querySelector('#error');


  // Presets Data Cache
  let presetsData = {};

  // Fetch sample presets from API
  fetch('/api/presets')
    .then(res => res.json())
    .then(data => {
      if (data.presets) {
        data.presets.forEach(p => {
          presetsData[p.id] = p.data;
        });
      }
    })
    .catch(err => console.warn('Could not load presets:', err));

  // Quick Preset Click Handler
  document.querySelectorAll('.preset-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const presetId = chip.dataset.preset;
      const data = presetsData[presetId];
      if (data) {
        Object.keys(data).forEach(key => {
          const input = form.querySelector(`[name="${key}"]`);
          if (input) {
            input.value = data[key];
            input.dispatchEvent(new Event('change'));
          }
        });
        // Flash button indication
        chip.style.transform = 'scale(0.95)';
        setTimeout(() => chip.style.transform = '', 150);
      }
    });
  });

  // Tab Switch Handler
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.dataset.tab;
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.querySelector(`#${targetTab}`).classList.add('active');
    });
  });

  // Switch to specific tab helper
  function switchTab(tabId) {
    const btn = document.querySelector(`[data-tab="${tabId}"]`);
    if (btn) btn.click();
  }

  // Clear Form Handler
  clearFormBtn.addEventListener('click', () => {
    form.reset();
    outputState.hidden = true;
    errorState.hidden = true;
    idleState.hidden = false;
  });

  // Helper to extract payload
  function getPayload() {
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());
    if (!payload.new_price || payload.new_price.trim() === '') {
      delete payload.new_price;
    }
    return payload;
  }

  // Submit Single Prediction
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    switchTab('tab-estimate');

    errorState.hidden = true;
    outputState.hidden = true;
    idleState.hidden = true;
    loadingState.hidden = false;
    submitSingleBtn.disabled = true;

    try {
      const payload = getPayload();
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Failed to calculate estimate');

      const price = Number(result.prediction_lakh);
      const minRange = Math.max(0, price * 0.9).toFixed(2);
      const maxRange = (price * 1.1).toFixed(2);

      document.querySelector('#price-main').innerHTML = `₹${price.toFixed(2)} <small>lakh</small>`;
      document.querySelector('#price-range-text').textContent = `₹${minRange} – ₹${maxRange} L`;
      document.querySelector('#used-model-name').textContent = result.model.replaceAll('_', ' ').toUpperCase();

      const meta = window.APP_METADATA || {};
      const score = (meta.metrics && meta.metrics[result.model]) ? meta.metrics[result.model].R2 : 'N/A';
      document.querySelector('#used-model-r2').textContent = score;

      // Estimated depreciation vs new price if provided
      if (payload.new_price) {
        const orig = parseFloat(payload.new_price);
        const dep = Math.max(0, ((orig - price) / orig) * 100).toFixed(1);
        document.querySelector('#est-depreciation').textContent = `${dep}% loss from new`;
      } else {
        document.querySelector('#est-depreciation').textContent = 'N/A (New price unlisted)';
      }

      document.querySelector('#result-confidence-text').textContent = result.confidence_note;
      outputState.hidden = false;

    } catch (err) {
      errorState.textContent = err.message;
      errorState.hidden = false;
      idleState.hidden = false;
    } finally {
      loadingState.hidden = true;
      submitSingleBtn.disabled = false;
    }
  });

  // Copy Valuation Summary Handler

  const copyBtn = document.querySelector('#copy-summary-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const priceText = document.querySelector('#price-main').textContent.trim();
      const rangeText = document.querySelector('#price-range-text').textContent.trim();
      const modelText = document.querySelector('#used-model-name').textContent.trim();
      const brand = document.querySelector('#brand').value || 'Vehicle';
      const year = document.querySelector('#year').value || '';

      const summary = `CarWorth AI Valuation Summary:\nVehicle: ${year} ${brand}\nPredicted Price: ${priceText}\nLikely Range: ${rangeText}\nModel Used: ${modelText}`;

      navigator.clipboard.writeText(summary).then(() => {
        const origText = copyBtn.innerHTML;
        copyBtn.textContent = 'Copied to Clipboard!';
        setTimeout(() => copyBtn.innerHTML = origText, 2000);
      });
    });
  }
});
