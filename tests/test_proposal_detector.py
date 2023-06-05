import unittest
from src.proposal_detector import KeywordProposalDetector

class TestKeywordProposalDetector(unittest.TestCase):
    def setUp(self):
        self.detector = KeywordProposalDetector()

    def test_is_proposal_with_empty_strings(self):
        title = ""
        document = ""
        result = self.detector.is_proposal(title, document)
        self.assertEqual(result, 0.0)

    def test_is_proposal_with_no_matching_keywords(self):
        title = "Some random title"
        document = "This is a document without any matching keywords"
        result = self.detector.is_proposal(title, document)
        self.assertEqual(result, 0.0)

    def test_is_proposal_with_only_title_matching_keywords(self):
        title = "Some random title"
        document = "This is a document with only title matching keywords: Some random title"
        result = self.detector.is_proposal(title, document)
        self.assertEqual(result, 0.6)

    def test_is_proposal_with_partial_matching_keywords(self):
        title = "Proposal for a project"
        document = "This is a document containing keywords like project and objective"
        result = self.detector.is_proposal(title, document)
        self.assertAlmostEqual(result, 0.344, places=3)

    def test_is_proposal_with_all_matching_keywords(self):
        title = "Random title"
        document = "This is a document with all the keywords and title random title: proposal, scope, objective, budget, deliverable, milestone"
        result = self.detector.is_proposal(title, document)
        self.assertEqual(result, 1)

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()

