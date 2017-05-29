var app = require('./index.js')
app.logging.config('logging', false, app.logging.log_level.INFO, 1024)
app.logging.info("Starting training process...")
window.training = new app.training(3000)
