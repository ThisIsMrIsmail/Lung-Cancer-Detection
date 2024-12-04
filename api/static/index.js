
const form = document.querySelector("form")
const image = document.querySelector("input")
const button = document.querySelector("button")
const pred = document.querySelector("#pred")
const nothing = document.querySelector("#nothing")
const loading = document.querySelector("#loading")

if (pred.textContent.length > 4)
    nothing.style.display = "none"

form.addEventListener("submit", (e) => {
    if (!image.files.length) {
        e.preventDefault()
        alert("Please, upload image to predict")
    } else {
        pred.style.display = "none"
        loading.style.display = "flex"
    }
})