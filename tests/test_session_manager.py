import sys
import os
import unittest
import shutil
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.session_manager import SessionManager

class TestSessionManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/data/sessions"
        self.manager = SessionManager(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_session(self):
        session_id = self.manager.create_new_session()
        self.assertIsNotNone(session_id)

    def test_save_and_load_session(self):
        session_id = "test_session"
        messages = [{"role": "user", "content": "hello"}]
        self.manager.save_session(session_id, messages, "Test Session")
        
        loaded = self.manager.load_session(session_id)
        self.assertEqual(loaded["session_id"], session_id)
        self.assertEqual(loaded["name"], "Test Session")
        self.assertEqual(loaded["messages"], messages)

    def test_list_sessions(self):
        self.manager.save_session("s1", [], "Session 1")
        self.manager.save_session("s2", [], "Session 2")
        
        sessions = self.manager.list_sessions()
        self.assertEqual(len(sessions), 2)
        names = [s["name"] for s in sessions]
        self.assertIn("Session 1", names)
        self.assertIn("Session 2", names)

if __name__ == '__main__':
    unittest.main()
