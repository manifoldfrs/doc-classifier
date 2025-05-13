#!/usr/bin/env python
"""
Quick test script for hitting the standard GET endpoints of the local API.

Usage:
  python scripts/test_get_endpoints.py
"""
import os
import sys
from typing import List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Ensure the script can find modules in the 'src' directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables (e.g., for PORT if customized)
load_dotenv()

BASE_URL = f"http://localhost:{os.getenv('PORT', '8000')}"

# Read API key from environment
ALLOWED_API_KEYS_STR = os.getenv("ALLOWED_API_KEYS", "")
API_KEYS = [key.strip() for key in ALLOWED_API_KEYS_STR.split(",") if key.strip()]
API_KEY = API_KEYS[0] if API_KEYS else None

# Define endpoints and whether they require authentication
# Format: (path, expected_status, requires_auth)
ENDPOINTS: List[Tuple[str, int, bool]] = [
    ("/v1/health", 200, True),
    ("/v1/version", 200, True),
    ("/metrics", 200, False),  # Assumes PROMETHEUS_ENABLED=true
    ("/docs", 200, False),  # Swagger UI HTML
    ("/openapi.json", 200, False),  # OpenAPI spec
]


def test_endpoint(
    endpoint: str, expected_status: int, requires_auth: bool, api_key: Optional[str]
) -> bool:
    """Tests a single GET endpoint, adding auth header if needed."""
    url = f"{BASE_URL}{endpoint}"
    test_passed = False
    headers = {}

    if requires_auth:
        if not api_key:
            print(f"--- Testing {endpoint} ---")
            print("⚠️ SKIPPED: Endpoint requires auth, but no API_KEY found in .env")
            print("-" * (len(endpoint) + 20))
            return False  # Consider skipped tests as not passed
        headers["x-api-key"] = api_key

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"--- Testing {endpoint} ---")
        print(f"Status Code: {response.status_code}")

        # Preview content, replacing newlines for better readability in a single line if needed
        content_preview = response.text[:150].replace("\n", " ") + (
            "..." if len(response.text) > 150 else ""
        )
        print(f"Response Preview: {content_preview}\n")

        if response.status_code == expected_status:
            print(f"✅ PASSED: Expected status {expected_status}")
            test_passed = True
        else:
            print(
                f"❌ FAILED: Expected status {expected_status}, got {response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        print(f"--- Testing {endpoint} ---")
        print(f"❌ FAILED: Could not connect to {url}. Is the server running?")
    except requests.exceptions.Timeout:
        print(f"--- Testing {endpoint} ---")
        print(f"❌ FAILED: Request timed out for {url}.")
    except Exception as e:
        print(f"--- Testing {endpoint} ---")
        print(f"❌ FAILED: An unexpected error occurred: {e}")

    print("-" * (len(endpoint) + 20))  # Separator
    return test_passed


def main():
    """Runs tests for all defined endpoints."""
    print(f"🚀 Starting GET endpoint tests against {BASE_URL}...")
    if API_KEY:
        print(f"🔑 Using API Key starting with: {API_KEY[:4]}...\n")
    else:
        print("⚠️ No API Key found in .env, authenticated endpoints will be skipped.\n")

    results = [
        test_endpoint(ep, status, req_auth, API_KEY)
        for ep, status, req_auth in ENDPOINTS
    ]

    total_tests = len(ENDPOINTS)
    # Count only tests that were actually run (not skipped)
    # We'll consider skipped tests as failures for the exit code
    run_results = [
        r
        for r, (ep, st, ra) in zip(results, ENDPOINTS, strict=False)
        if not (ra and not API_KEY)
    ]
    passed_tests = sum(run_results)
    failed_tests = total_tests - passed_tests  # Includes skipped tests as failures

    print("\n--- Test Summary ---")
    print(f"Total Endpoints Defined: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print("-" * 20)

    if failed_tests > 0:
        print("🔥 Some GET endpoint tests failed or were skipped.")
        sys.exit(1)  # Exit with non-zero code if any tests failed
    else:
        print("🎉 All GET endpoint tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
