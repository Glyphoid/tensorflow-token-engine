#!/usr/bin/python

import unittest

import sys
import concurrent.futures
import requests
import json
import argparse

test_config = {
    "server_url": "http://127.0.0.1/glyphoid/token-recog"
}

class TestTokenServer(unittest.TestCase):
    # Member variables #
    headers = {"content-type": "application/json"}

    # Private methods #
    def _send_request_to_url_get_response(self, url, post_json_body):
        return requests.post(url,
                             headers=self.headers,
                             data=json.dumps(post_json_body))

    def _send_request_get_response(self, post_json_body):
        return self._send_request_to_url_get_response(test_config["server_url"], post_json_body)

    def _send_request_get_json(self, post_json_body):
        response = self._send_request_get_response(post_json_body)
        return response.json()

    def _send_request_get_winner(self, post_json_body):
        response_json = self._send_request_get_json(post_json_body)

        return response_json["winnerTokenName"]

    # Public (test) methods #

    # Invalid json body: Missing feature vector
    def test_missing_feature_vector(self):
        post_body = {}
        response = self._send_request_get_response(post_body)

        self.assertEqual(406, response.status_code)
        self.assertEqual(u'Missing featureVector field from JSON request body', response.json()["error"])

    def test_invalid_feature_vector(self):
        post_body = {"featureVector": [-1]}
        response = self._send_request_get_response(post_body)

        self.assertEqual(406, response.status_code)
        self.assertEqual(u'Invalid feature vector in request', response.json()["error"])

    def test_invalid_path(self):
        post_body = {}
        response = self._send_request_to_url_get_response(test_config["server_url"] + "/foo", post_body)

        self.assertEqual(404, response.status_code)
        self.assertEqual(u'Invalid path', response.json()["error"])

    # Concurrent valid requests
    def test_concurrent_valid_requests(self):
        n_tasks = 40
        futures = [None] * n_tasks

        correct_token = "gr_io"
        post_body = {"featureVector": [-0.357914567, -0.360252172, -0.220175385, -0.46692428, 1.571988225, 1.813607812,
                                       1.726060033, 1.65583539, 1.590689898, 1.631353378, 1.472156048, 0.38140282,
                                       -0.45616135, -0.662442923, -1.008334756, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0.123000003, 1, 0.813000023, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.883453965, 1]}

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_tasks) as executor:
            for i in range(n_tasks):
                futures[i] = executor.submit(self._send_request_get_winner, post_body)

            for i in range(n_tasks):
                self.assertEqual(correct_token, futures[i].result())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", dest='server_url')
    parser.add_argument('unittest_args', nargs='*')

    # Parse and apply optional arguments
    args = vars(parser.parse_args())
    for key in test_config:
        if args[key]:
            print "Overriding test config item %s: " % key, test_config[key], "->", args[key]
            test_config[key] = args[key]

    sys.argv[1:] = args["unittest_args"]

    unittest.main()
