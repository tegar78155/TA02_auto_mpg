document.querySelectorAll("[data-weight]").forEach((btn) => {
    btn.addEventListener("click", () => {
        const input = document.getElementById("weight");
        if (input) input.value = btn.dataset.weight;
    });
});

const form = document.getElementById("predictForm");
const btn = document.getElementById("submitBtn");
const spinner = document.getElementById("spinner");
if (form && btn && spinner) {
    form.addEventListener("submit", () => {
        btn.disabled = true;
        spinner.classList.remove("d-none");
    });
}
