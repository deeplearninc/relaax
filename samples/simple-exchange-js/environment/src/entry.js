var app = require('./index.js')
app.logging.config('logging')
app.logging.info("Starting training process...")
window.training = new app.training(11)
