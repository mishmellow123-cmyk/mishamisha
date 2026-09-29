"""A job's "branch" reaches every unit, and the node agent fetches it (it used to fetch claude/long-dawn-v2 whatever
the job said, so a render ran code the job was not written against). The name reaches a shell on the node, so both
ends refuse anything that is not a plain branch name.

Run: python -m pytest -q the-long-dawn/cloud/tests/test_farm_branch.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import farm  # noqa: E402
import farm_node  # noqa: E402

BAD = ('x; rm -rf ~', '../main', 'a..b', '-upload-pack=x', 'b a', '$(id)', 'x`id`', '', 'x\ny')


class FarmBranchTests(unittest.TestCase):
    def test_the_daemon_passes_the_jobs_branch(self):
        self.assertEqual(farm.job_branch({'branch': 'claude/owner-night-20260929'}), 'claude/owner-night-20260929')
        self.assertIsNone(farm.job_branch({}))

    def test_the_daemon_refuses_a_bad_branch(self):
        for b in BAD:
            with self.subTest(b=b), self.assertRaises(SystemExit):
                farm.job_branch({'name': 't', 'branch': b})

    def test_the_agent_fetches_the_units_branch_else_its_default(self):
        self.assertEqual(farm_node.unit_branch({'branch': 'claude/owner-night-20260929'}), 'claude/owner-night-20260929')
        self.assertEqual(farm_node.unit_branch({}), farm_node.BRANCH)
        self.assertEqual(farm_node.unit_branch({'branch': None}), farm_node.BRANCH)

    def test_the_agent_refuses_a_bad_branch(self):
        for b in BAD[:-2]:                       # '' falls back to the default, as a missing key does
            with self.subTest(b=b), self.assertRaises(RuntimeError):
                farm_node.unit_branch({'branch': b})

    def test_every_unit_carries_the_branch(self):
        src = open(farm.__file__).read()
        self.assertIn('branch=job_branch(self.spec)', src)
        self.assertEqual(src.count('def _unit('), 1)


if __name__ == '__main__':
    unittest.main()
