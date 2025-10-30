// 🌙 DARK MODE TOGGLE
const btn = document.getElementById("darkModeBtn");
btn.addEventListener("click", () => {
  document.body.classList.toggle("dark-mode");
  btn.textContent = document.body.classList.contains("dark-mode")
    ? "☀️ Light Mode"
    : "🌙 Dark Mode";
});

// 💹 LIVE PRICE FETCH (updates every 10s)
async function updateLivePrice() {
  if (!ticker) return;
  try {
    const res = await fetch(`/live_price/${ticker}`);
    const data = await res.json();
    document.getElementById("livePrice").textContent = `$${data.price.toFixed(2)}`;
  } catch (err) {
    document.getElementById("livePrice").textContent = "Error fetching price";
  }
}
setInterval(updateLivePrice, 10000);
updateLivePrice();
