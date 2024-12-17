import os
import subprocess
import pytest
import csv
import numpy as np
from unittest.mock import mock_open, patch
from predator_prey import simulate_predator_prey as spp
from argparse import ArgumentTypeError

# Constants for directory paths used in tests
DATA_DIR = './landscapes'
RESULT_DIR = './'
CSV_FILE = "averages.csv"

# Predefined invalid file errors to be confirmed during testing
INVALID_FILES_ERRORS = spp.INVALID_FILES_ERRORS

# Expected results directly in the pytest file
EXPECTED_RESULTS = {
    # timestep -> (time, mice_density, fox_density)
    "10x20.dat": {
        0: (0.0,2.42685957886714565,2.42685957886714565),
        150: (75.0,1.93132686050462898,2.03811390783230584),
        300: (150.0,1.94081293427889712,1.54153299080919104),
        450: (225.0,2.30450296460307724,1.18109759878066334),
        600: (300.0,3.02527122801705994,0.98386929513174370),
        750: (375.0,4.09839487004793757,0.93434587945126768),
        900: (450.0,5.40999958623586608,1.00781561219744309),
        990: (495.0,1.07479697405524144,1.12876813197949777),
    },
    "1x1.dat": {
        0: (0.0,0.67182122056200611,0.67182122056200611),
        150: (75.0,0.31052043365706333,0.89909285362623748),
        300: (150.0,0.11458244075447316,2.13914919443749074),
        450: (225.0,1.50381544629426789,10.46110608672806386),
        600: (300.0,2.95292663650967580,0.03104226753509893),
        750: (375.0,0.02772411131841630,0.22782446796352190),
        900: (450.0,26.58762755608572448,4.63964385444049476),
        990: (495.0,0.00066433633753296,0.47257002350616306),
    },
    "1x1land.dat": {
        0: (0.0,0.67182122056200611,0.67182122056200611),
        150: (75.0,0.31052043365706333,0.89909285362623748),
        300: (150.0,0.11458244075447316,2.13914919443749074),
        450: (225.0,1.50381544629426789,10.46110608672806386),
        600: (300.0,2.95292663650967580,0.03104226753509893),
        750: (375.0,0.02772411131841630,0.22782446796352190),
        900: (450.0,26.58762755608572448,4.63964385444049476),
        990: (495.0,0.00066433633753296,0.47257002350616306),
    },
    "1x1water.dat": {
        0: (0.0,0.00000000000000000,0.00000000000000000),
        150: (75.0,0.00000000000000000,0.00000000000000000),
        300: (150.0,0.00000000000000000,0.00000000000000000),
        450: (225.0,0.00000000000000000,0.00000000000000000),
        600: (300.0,0.00000000000000000,0.00000000000000000),
        750: (375.0,0.00000000000000000,0.00000000000000000),
        900: (450.0,0.00000000000000000,0.00000000000000000),
        990: (495.0,0.00000000000000000,0.00000000000000000),
    },
    "1x50.dat": {
        0: (0.0,2.37978269038848289,2.37978269038848289),
        150: (75.0,1.86373763682497451,2.01349806759624350),
        300: (150.0,1.90550457856713473,1.51238859521130942),
        450: (225.0,2.28098265444406811,1.15331241827840336),
        600: (300.0,2.97067402647181034,0.95155873259759394),
        750: (375.0,3.95455832444631916,0.87850541820831263),
        900: (450.0,5.11637163935770101,0.88914438550346631),
        990: (495.0,0.95615878697252366,1.22541147589595822),
    },
    "20x20corner.dat": {
        0: (0.0,2.45477707708733517,2.45477707708733517),
        150: (75.0,1.93458313330523435,2.07242744444466886),
        300: (150.0,1.91867200467404531,1.56783345128020302),
        450: (225.0,2.25642368711042840,1.19469712133643191),
        600: (300.0,2.94621136613776802,0.98425437465501420),
        750: (375.0,3.98449553799031708,0.91940534484229441),
        900: (450.0,5.26626246129534437,0.97178867185688511),
        990: (495.0,1.05044256153286297,1.16492587500587530),
    },
    "20x20land.dat": {
        0: (0.0,2.44615582248251506,2.44615582248251506),
        150: (75.0,1.93830844234003163,2.06084225802753407),
        300: (150.0,1.93250274421262369,1.56043582923605517),
        450: (225.0,2.28132338271657709,1.19302797371095970),
        600: (300.0,2.98604349538463643,0.98906311397119340),
        750: (375.0,4.04443130849182442,0.93351036938730081),
        900: (450.0,5.34933850188576088,1.00238206700264021),
        990: (495.0,1.07675010606558841,1.14357273840052120),
    },
    "20x20water.dat": {
        0: (0.0,0.00000000000000000,0.00000000000000000),
        150: (75.0,0.00000000000000000,0.00000000000000000),
        300: (150.0,0.00000000000000000,0.00000000000000000),
        450: (225.0,0.00000000000000000,0.00000000000000000),
        600: (300.0,0.00000000000000000,0.00000000000000000),
        750: (375.0,0.00000000000000000,0.00000000000000000),
        900: (450.0,0.00000000000000000,0.00000000000000000),
        990: (495.0,0.00000000000000000,0.00000000000000000),
    },
    "3x1.dat": {
        0: (0.0,2.45449495262408490,2.45449495262408490),
        150: (75.0,1.56471439513624611,2.97438392316747624),
        300: (150.0,1.18956378235056581,3.91666444781847289),
        450: (225.0,2.50152966255910414,8.54449747229258705),
        600: (300.0,7.36186057292365348,1.42067848343779768),
        750: (375.0,2.94879521815449275,0.19308541680438979),
        900: (450.0,13.64368221163453931,2.37271435759072080),
        990: (495.0,0.60979661819400477,6.09407775826875575),
    },
    "3x3.dat": {
        0: (0.0,2.48874649348825550,2.48874649348825550),
        150: (75.0,1.94351518129473844,2.11721837946698965),
        300: (150.0,1.89506781924912415,1.60368112034451071),
        450: (225.0,2.20063777454958931,1.21481289022659178),
        600: (300.0,2.85290610530400413,0.98827176857611176),
        750: (375.0,3.84940644014201361,0.90602753023024485),
        900: (450.0,5.09508980849616400,0.93665688170406858),
        990: (495.0,1.02775712811074360,1.20984017244028141),
    },
    "40x20ps.dat": {
        0: (0.0,2.51308728990640295,2.51308728990640295),
        150: (75.0,1.92800491452653744,2.14250814681620927),
        300: (150.0,1.86098085564083671,1.61809269609257411),
        450: (225.0,2.14268162807601259,1.21670179885501306),
        600: (300.0,2.75613158132590108,0.97598807457075021),
        750: (375.0,3.69107438442805158,0.87293757662925231),
        900: (450.0,4.84928196195184125,0.86776417936137928),
        990: (495.0,0.96261581450970579,1.28209126666458162),
    },
    "50x1.dat": {
        0: (0.0,2.37978269038848289,2.37978269038848289),
        150: (75.0,1.86373763682497451,2.01349806759624350),
        300: (150.0,1.90550457856713473,1.51238859521130964),
        450: (225.0,2.28098265444406811,1.15331241827840336),
        600: (300.0,2.97067402647181078,0.95155873259759371),
        750: (375.0,3.95455832444631916,0.87850541820831263),
        900: (450.0,5.11637163935770101,0.88914438550346631),
        990: (495.0,0.95615878697252388,1.22541147589595822),
    },
    "map.dat": {
        0: (0.0,2.42685957886714565,2.42685957886714565),
        150: (75.0,1.93132686050462898,2.03811390783230584),
        300: (150.0,1.94081293427889712,1.54153299080919104),
        450: (225.0,2.30450296460307724,1.18109759878066334),
        600: (300.0,3.02527122801705994,0.98386929513174370),
        750: (375.0,4.09839487004793757,0.93434587945126768),
        900: (450.0,5.40999958623586608,1.00781561219744309),
        990: (495.0,1.07479697405524144,1.12876813197949777),
    },
    "test.dat": {
        0: (0.0,2.55016303271148237,2.55016303271148237),
        150: (75.0,1.94247352481352853,2.19087394602230079),
        300: (150.0,1.84341489660960955,1.65885323762601500),
        450: (225.0,2.09485792838773666,1.24241373633161123),
        600: (300.0,2.67540338256194632,0.98705453669387089),
        750: (375.0,3.57640592103804300,0.87075613121357165),
        900: (450.0,4.70849845273872791,0.85314600639689719),
        990: (495.0,0.95653421836307573,1.32388519178843134),
    },
    "two_square_islands.dat": {
        0: (0.0,2.45463804083453541,2.45463804083453541),
        150: (75.0,1.93437860519444604,2.07541240875557209),
        300: (150.0,1.91624333180365780,1.57039028047067619),
        450: (225.0,2.25172650898930771,1.19559344060314854),
        600: (300.0,2.93645229385174389,0.98315091796499132),
        750: (375.0,3.96229838780324073,0.91555566149031886),
        900: (450.0,5.21861649206427103,0.96258252107157793),
        990: (495.0,1.03555942896753828,1.18341996418087825),
    }
}

# uni-test

# Test to verify the correct version number is retrieved from the module
def test_get_version():
    assert spp.getVersion() == 3.0

# Validator Tests
# Test for checking positive float validation
def test_check_positive_float():
    assert spp.check_positive_float("0.1") == 0.1
    with pytest.raises(ArgumentTypeError):
        spp.check_positive_float("0")
    with pytest.raises(ArgumentTypeError):
        spp.check_positive_float("-1")
    with pytest.raises(ValueError):
        spp.check_positive_float("not_a_float")

def test_check_positive_int():
    assert spp.check_positive_int("1") == 1
    with pytest.raises(ArgumentTypeError):
        spp.check_positive_int("0")
    with pytest.raises(ArgumentTypeError):
        spp.check_positive_int("-1")
    with pytest.raises(ValueError):
        spp.check_positive_int("3.5")

# Argument Parsing Test
def test_parse_arguments():
    # Simulate valid command-line arguments
    with patch('sys.argv', [
        'program', '-r', '0.1', '-a', '0.05', '-k', '0.2', '-b', '0.03',
        '-m', '0.09', '-l', '0.2', '-dt', '0.5', '-t', '10', '-d', '500',
        '-f', 'landscape.dat', '-ms', '1', '-fs', '1'
    ]):
        args = spp.parse_arguments()
        assert args.birth_mice == 0.1
    # Test missing required arguments
    with patch('sys.argv', ['program']):
        with pytest.raises(SystemExit):
            spp.parse_arguments()

# Landscape File Tests
@pytest.mark.parametrize("file_name, error_message", INVALID_FILES_ERRORS.items())
def test_invalid_input_files(file_name, error_message):
    with patch("os.path.exists", return_value=True), \
            patch("builtins.open", mock_open(read_data="")):
        with pytest.raises(RuntimeError, match=error_message):
            spp.read_landscape(file_name)

# Test reading of landscape file to verify correct dimensions and content parsing
def test_read_landscape():
    mock_data = "3 3\n1 1 1\n1 0 1\n1 1 1\n"
    with patch("builtins.open", mock_open(read_data=mock_data)), \
         patch("os.path.exists", return_value=True):
        landscape, width, height = spp.read_landscape("dummy.dat")
        assert height == 3
        assert width == 3

    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            spp.read_landscape("nonexistent.dat")

# Density Initialization Test
def test_initialize_density():
    landscape = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    density = spp.initialize_density(landscape, 1, 1, 1)
    assert density[1, 1] != 0

# CSV File and Average Calculation Tests
def test_write_csv_header():
    with patch("builtins.open", mock_open()) as mock_file:
        spp.write_csv_header()
        mock_file.assert_called_once_with(CSV_FILE, "w")
        mock_file().write.assert_called_once_with("Timestep,Time,Mice,Foxes\n")

def test_calculate_average_density():
    density = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
    assert spp.calculate_average_density(density, 3) == 1
    assert spp.calculate_average_density(density, 0) == 0
    
# Fixtures
@pytest.fixture(scope="module")
def run_simulation():
    '''
    Fixture to execute simulation subprocess with required arguments.
    '''
    def _run(dat_file):
        return subprocess.run(
            [
                "python", "-m", "predator_prey.simulate_predator_prey",
                "-r", "0.1", "-a", "0.05", "-k", "0.2",
                "-b", "0.03", "-m", "0.09", "-l", "0.2",
                "-dt", "0.5", "-t", "10", "-d", "500",
                "-f", dat_file, "-ms", "1", "-fs", "1"
            ],
            capture_output=True, text=True
        )
    return _run

# Simulation Tests
@pytest.mark.parametrize("dat_file", EXPECTED_RESULTS.keys())
def test_valid_simulation_output(dat_file, run_simulation):
    file_path = os.path.join(DATA_DIR, dat_file)
    result = run_simulation(file_path)
    assert result.returncode == 0, f"Simulation execution failed for '{dat_file}'"

    result_output_file = os.path.join(RESULT_DIR, CSV_FILE)
    assert os.path.exists(result_output_file), f"Output file not found for '{dat_file}'"

    with open(result_output_file, 'r') as output_file:
        reader = csv.reader(output_file)
        headers = next(reader)
        assert headers == ["Timestep", "Time", "Mice", "Foxes"], f"Header mismatch for '{dat_file}'"

        for row in reader:
            timestep, time, mice, foxes = map(float, row)
            expected = EXPECTED_RESULTS[dat_file].get(int(timestep))
            if expected:
                expected_time, expected_mice, expected_foxes = expected
                assert abs(time - expected_time) < 1e-9, f"Time mismatch at step {timestep} for '{dat_file}'"
                assert abs(mice - expected_mice) < 1e-9, f"Mice count mismatch at step {timestep} for '{dat_file}'"
                assert abs(foxes - expected_foxes) < 1e-9, f"Fox count mismatch at step {timestep} for '{dat_file}'"