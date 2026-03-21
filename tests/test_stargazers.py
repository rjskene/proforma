# SPDX-FileCopyrightText: 2024-present rjskene <rjskene83@gmail.com>
#
# SPDX-License-Identifier: MIT

import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from proforma.stargazers import who_starred


class TestWhoStarred(unittest.TestCase):
    def _make_response(self, data):
        """Return a mock urllib response whose read() yields JSON-encoded *data*."""
        response = MagicMock()
        response.read.return_value = json.dumps(data).encode()
        response.__enter__ = lambda s: s
        response.__exit__ = MagicMock(return_value=False)
        return response

    @patch("proforma.stargazers.urllib.request.urlopen")
    def test_returns_list_of_logins(self, mock_urlopen):
        """Returns the login names from a single page of stargazers."""
        page1 = [{"login": "alice"}, {"login": "bob"}]
        page2 = []
        mock_urlopen.side_effect = [
            self._make_response(page1),
            self._make_response(page2),
        ]

        result = who_starred(owner="rjskene", repo="proforma")

        self.assertEqual(result, ["alice", "bob"])

    @patch("proforma.stargazers.urllib.request.urlopen")
    def test_paginates_multiple_pages(self, mock_urlopen):
        """Collects stargazers across multiple API pages."""
        page1 = [{"login": f"user{i}"} for i in range(100)]
        page2 = [{"login": "last_user"}]
        page3 = []
        mock_urlopen.side_effect = [
            self._make_response(page1),
            self._make_response(page2),
            self._make_response(page3),
        ]

        result = who_starred(owner="rjskene", repo="proforma")

        self.assertEqual(len(result), 101)
        self.assertEqual(result[-1], "last_user")

    @patch("proforma.stargazers.urllib.request.urlopen")
    def test_empty_repo_returns_empty_list(self, mock_urlopen):
        """Returns an empty list when no one has starred the repository."""
        mock_urlopen.side_effect = [self._make_response([])]

        result = who_starred(owner="rjskene", repo="proforma")

        self.assertEqual(result, [])

    @patch("proforma.stargazers.urllib.request.urlopen")
    def test_token_added_to_authorization_header(self, mock_urlopen):
        """Passes the token as an Authorization header when provided."""
        mock_urlopen.side_effect = [self._make_response([])]

        who_starred(owner="rjskene", repo="proforma", token="my_secret_token")

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertEqual(
            request_obj.get_header("Authorization"), "Bearer my_secret_token"
        )

    @patch("proforma.stargazers.urllib.request.urlopen")
    def test_no_token_omits_authorization_header(self, mock_urlopen):
        """Does not set an Authorization header when no token is provided."""
        mock_urlopen.side_effect = [self._make_response([])]

        who_starred(owner="rjskene", repo="proforma")

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertIsNone(request_obj.get_header("Authorization"))

    def test_who_starred_importable_from_top_level(self):
        """who_starred is accessible via the top-level proforma package."""
        import proforma

        self.assertTrue(callable(proforma.who_starred))
        self.assertIn("who_starred", proforma.__all__)


if __name__ == "__main__":
    unittest.main()
