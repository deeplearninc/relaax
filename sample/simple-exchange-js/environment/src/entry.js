var app = require('./index.js')
app.logging.config(logging_element_id='logging')
app.logging.info("Starting training process...")
window.training = new app.training()
