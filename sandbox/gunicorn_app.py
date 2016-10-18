def main(n_worker=0):
    def app(environ, start_response):
        data = "%s Hello, World!\n" % repr(n_worker)
        start_response("200 OK", [
            ("Content-Type", "text/plain"),
            ("Content-Length", str(len(data)))
            ])
        return iter([data])
    return app
