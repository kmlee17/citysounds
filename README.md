## CitySounds
### Audio Classes (8732 samples)
* Air Conditioner
* Car Horn
* Children Playing
* Dog Bark
* Drilling
* Enginge Idling
* Gun Shot
* Jackhammer
* Siren
* Street Music

### Packages Used
* Pandas
* Numpy
* Scikit-Learn
* Scipy
* Librosa
* Flask
* Matplotlib
* Seaborn
* SoX (command-line audio processing)
* Aubio

### Grid Search
* GradientBoost: {n_estimators=100, learning_rate=0.1, max_depth=5}
	- 0.722968300000
* RandomForest: {'n_estimators': 250, 'criterion': 'gini'}
	- 0.730531378836
* KNearestNeighbors: {n_neighbors=34}
	- 0.722056802565
* SVM: {'kernel': 'rbf', 'C': 0.3, 'gamma': 0.05}
	- 0.744388456253

## Credits

Original dataset created by Justin Salamon, Christopher Jacoby, and Juan Pablo Bello

Music and Audio Research Lab (MARL), New York University, USA Center for Urban Science and Progress (CUSP), New York University, USA

* http://serv.cusp.nyu.edu/projects/urbansounddataset
* http://marl.smusic.nyu.edu/
* http://cusp.nyu.edu/

Web app template created by and is maintained by **David Miller**, Managing Parter at [Iron Summit Media Strategies](http://www.ironsummitmedia.com/).

* https://twitter.com/davidmillerskt
* https://github.com/davidtmiller

Start Bootstrap is based on the [Bootstrap](http://getbootstrap.com/) framework created by [Mark Otto](https://twitter.com/mdo) and [Jacob Thorton](https://twitter.com/fat).