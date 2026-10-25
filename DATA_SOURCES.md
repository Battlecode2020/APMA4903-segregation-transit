# Data sources and attribution

This public snapshot contains aggregate Census counts, derived travel estimates, geographic neighbor graphs, maps and analysis outputs. Raw agency GTFS archives and raw Census downloads remain outside the repository.

The source acquisition and parsing code is in `scripts/multicity_census.py` and `scripts/multicity_sources.py`. The published `docs/outputs/multicity/transit_sources.json` records producers, download/archive URLs, schedule dates, agency coverage and SHA256 checksums. The original snapshot was calculated locally without Routes API calls.

## Required agency notices

SFMTA required derivative-data notice:
Reproduced with permission granted by the City and County of San Francisco.  The information has been provided by means of a nonexclusive, limited, and revocable license granted by the City and County of San Francisco.

The City and County of San Francisco does not guarantee the accuracy, adequacy, completeness or usefulness of any information.  The City and County of San Francisco provides this information "as is," without warranty of any kind, express or implied, including but not limited to warranties of merchantability or fitness for a particular purpose, and assumes no responsibility for anyone's use of the information.

Route and departure data provided by permission of UTA.

Contains information from City of Phoenix / Valley Metro GTFS, made available under the ODC Attribution License.


## Geographic segregation references

See [author-code references and methodology decisions](docs/data/references/fryer/citations.json). The repository does not redistribute the paper or claim agency endorsement. Source-dataset rights and terms remain with their respective providers.
