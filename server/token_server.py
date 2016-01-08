#!/usr/bin/env python

import os
import sys
import exceptions
import traceback

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import server_utils

import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from token_recognizer import TokenRecognizer


# Error types definition
class RequestPathError(exceptions.AttributeError):
    pass

class RequestJsonParsingError(exceptions.ValueError):
    pass

class RequestContentError(exceptions.AttributeError):
    pass

class FeatureVectorError(exceptions.AttributeError):
    pass

# Default values
config = {
    "serverPath": "/glyphoid/token-recog",
    "errorKey": "error",
    "maxNumCandidates": 10,
    "agentName": "TensorFlowPythonTokenRecognizer"
}

class TokenRecogServer(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def _set_success(self):
        self.send_response(200)
        self._set_headers()

    def _set_invalid_path(self):
        self.send_response(404)
        self._set_headers()
        self.wfile.write(json.dumps({config["errorKey"]: "Invalid path"}))

    def _set_invalid_request(self):
        self.send_response(406)
        self._set_headers()
        self.wfile.write(json.dumps({config["errorKey"]: "Invalid JSON in request body"}))

    def _set_missing_feature_vector(self):
        self.send_response(406)
        self._set_headers()
        self.wfile.write(json.dumps({config["errorKey"]: "Missing featureVector field from JSON request body"}))

    def _set_feature_vector_error(self):
        self.send_response(406)
        self._set_headers()
        self.wfile.write(json.dumps({config["errorKey"]: "Invalid feature vector in request"}))

    def _set_internal_server_error(self):
        self.send_response(500)
        self._set_headers()
        self.wfile.write(json.dumps({config["errorKey"]: "Internal server error"}))

    def do_POST(self):
        # Read request body string
        content_len = int(self.headers.getheader('content-length', 0))
        req_body_str = self.rfile.read(content_len)

        try:
            if self.path != config["serverPath"]:
                raise RequestPathError("Invalid path: \"%s\"" % self.path)

            try:
                req_body = json.loads(req_body_str)
            except ValueError:
                raise RequestJsonParsingError()

            if not "featureVector" in req_body:
                print "Missing featureVector"
                raise RequestContentError("Missing featureVector from request body");

            feature_vec = np.array(req_body["featureVector"])
            if len(np.shape(feature_vec)) == 1:
                feature_vec = np.array([feature_vec])
            # print "feature_vec = ", feature_vec     #DEBUG

            try:
                (winner_token_name, recog_ps) = tokenRecognizer.recognize(feature_vec)
            except ValueError:
                raise FeatureVectorError()

            winner_token_name = winner_token_name[0]
            recog_ps = recog_ps[0]

            recog_ps = sorted(recog_ps, key=lambda x : float(x[1]), reverse=True)
            recog_ps = recog_ps[: int(config["maxNumCandidates"])]

            # print "winner_token_name = ", winner_token_name #DEBUG
            # print "recog_ps = ", recog_ps #DEBUG

            resp_obj = {
                "agent": config["agentName"],
                "winnerTokenName": winner_token_name,
                "recogPVals": recog_ps
            }

            self._set_success()
            self.wfile.write(json.dumps(resp_obj))

        except RequestPathError:
            self._set_invalid_path()

        except RequestJsonParsingError:
            self._set_invalid_request()

        except RequestContentError:
            self._set_missing_feature_vector()

        except FeatureVectorError:
            self._set_feature_vector_error()

        except exceptions:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)

            self._set_internal_server_error()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: " + __file__ + " server_properties_path"
        sys.exit(1)

    # Read server properties file
    server_props = server_utils.read_properties_file(sys.argv[1])

    print "Server properties:"
    print repr(server_props)

    # Get instance of TokenRecognizer
    tokenRecognizer = TokenRecognizer(server_props["modelParamsPath"], 
                                      server_props["tokenNamesPath"])

    # Create instance of TokenRecogServer
    # token_recog_server = TokenRecogServer()

    port = int(server_props["serverPort"])
    server_address = ('', port)

    # Override server config
    for key in config:
        if key in server_props and config[key] != server_props[key]:
            print "Overriding property %s: " % key, config[key], "->", server_props[key]
            config[key] = server_props[key]

    httpd = HTTPServer(server_address, TokenRecogServer)

    print 'Starting token server httpd on port %d ...' % port
    httpd.serve_forever()
