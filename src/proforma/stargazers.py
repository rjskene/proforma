# SPDX-FileCopyrightText: 2024-present rjskene <rjskene83@gmail.com>
#
# SPDX-License-Identifier: MIT

import json
import urllib.request
from typing import List, Optional

GITHUB_API_URL = "https://api.github.com"
REPO_OWNER = "rjskene"
REPO_NAME = "proforma"


def who_starred(
    owner: str = REPO_OWNER,
    repo: str = REPO_NAME,
    token: Optional[str] = None,
) -> List[str]:
    """Fetch the list of GitHub users who have starred the repository.

    Parameters
    ----------
    owner : str
        The GitHub repository owner. Defaults to 'rjskene'.
    repo : str
        The GitHub repository name. Defaults to 'proforma'.
    token : str, optional
        A GitHub personal access token for authentication. Required for
        private repositories and useful to raise the API rate limit.

    Returns
    -------
    list of str
        A list of GitHub login names (usernames) that have starred the
        repository, in the order returned by the GitHub API.

    Raises
    ------
    urllib.error.HTTPError
        If the GitHub API request fails (e.g. repository not found or
        rate-limit exceeded).
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/stargazers"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "proforma-stargazers",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    stargazers: List[str] = []
    page = 1
    while True:
        paginated_url = f"{url}?per_page=100&page={page}"
        req = urllib.request.Request(paginated_url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())

        if not data:
            break

        stargazers.extend(user["login"] for user in data)
        page += 1

    return stargazers
