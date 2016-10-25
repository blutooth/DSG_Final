"""Submission creator."""
from zipfile import ZipFile


def make_submission(preds, fname='Y_test.predict'):
    """
    Make a submission.

    preds is an array of prediction probabilities.
    """
    with open(fname, 'w') as f:
        print "Creating submission, {} lines".format(len(preds))
        for pred in preds:
            f.write(str(pred) + '\n')

    with ZipFile(fname + '.zip', 'w') as zf:
        zf.write(fname)
