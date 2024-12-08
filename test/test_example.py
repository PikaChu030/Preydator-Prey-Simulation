from predator_prey import simulate_predator_prey

# pytest for get version
def testGetVersion():
    assert simulate_predator_prey.getVersion() == 3.0
