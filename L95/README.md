# L95: Parser Evaluation

These are the files associated with with the final assignment for L95.

The contents of the files are as follows:

## input.txt

Sentences used as input to the parsers

## gs.conll

CONLL file containing the Gold Standard used for evaluation

## gs_for_eval.conll

The Gold Standard file with some formatting changes needed for automated evaluation and with some sentences removed.

## nn_output.conll, sr_input.conll

Raw output of the Neural Network and Shift-Reduce Parsers, respectively.

## nn_output_for_eval.conll, sr_output_for_eval.conll

The parser output files with some formatting changes needed for automated evaluation and with some sentences removed.

## eval.xml

File given as input to MaltEval to control its output

## nn_results.txt, sr_results.txt

Output of MaltEval evaluation on each parser

## calc_averages.py

Python file to calculate micro and macro averages for each parser

## terminal_commands.txt

List of commands used to run parsers and MaltEval.
