const numberFormat = new Intl.NumberFormat();
const percentFormat = new Intl.NumberFormat(undefined, {
  style: "percent",
  maximumFractionDigits: 1,
});

function isCount(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function countLabel(value) {
  return isCount(value) ? numberFormat.format(value) : "Usage unavailable";
}

function percentLabel(fraction) {
  if (fraction > 0 && fraction < 0.001) return "<0.1%";
  if (fraction < 1 && fraction > 0.999) return ">99.9%";
  return percentFormat.format(fraction);
}

export function createContextMonitor(panel) {
  const find = (id) => panel.querySelector(`#${id}`);
  const peakLabel = find("context-peak");
  const limitLabel = find("context-limit");
  const meter = find("context-meter");
  const used = find("context-used");
  const remaining = find("context-remaining");
  const breakdown = find("context-breakdown");
  const usedPercent = find("used-percent");
  const remainingPercent = find("remaining-percent");
  const totalInput = find("total-input");
  const totalOutput = find("total-output");

  return function render(usage) {
    const peak = usage?.peak_input_tokens;
    const limit = usage?.input_token_limit;
    const hasLimit = isCount(limit) && limit > 0;
    const canMeasure = isCount(peak) && hasLimit;
    peakLabel.textContent = isCount(peak) ? `${countLabel(peak)} peak input tokens` : "Usage unavailable";
    limitLabel.textContent = hasLimit ? `${countLabel(limit)} input token limit` : "Limit unknown";
    totalInput.textContent = countLabel(usage?.input_tokens);
    totalOutput.textContent = countLabel(usage?.output_tokens);
    meter.hidden = !canMeasure;
    breakdown.hidden = !canMeasure;
    if (!canMeasure) return;

    const fraction = Math.min(peak / limit, 1);
    used.style.width = `${fraction * 100}%`;
    remaining.style.width = `${(1 - fraction) * 100}%`;
    usedPercent.textContent = `${percentLabel(peak / limit)} used`;
    remainingPercent.textContent = `${percentLabel(1 - fraction)} available`;
    meter.setAttribute("aria-valuenow", Math.min(peak, limit));
    meter.setAttribute("aria-valuemax", limit);
    meter.setAttribute("aria-valuetext", `${countLabel(peak)} of ${countLabel(limit)} input tokens`);
  };
}
