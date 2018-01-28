## глава 1:

### Связь, валидность, интерпретируемость коннектомики по анатомии их взаимно заменяемость/дополняемость:



1. Исследование их взаимосвязи и понимание сколько информации несет каждая?


	- Canonical correlation
	

	- regularized regression (Total Variation + L1 for example) predicting individual graph edges with vertex-level predictor variables



2. Какие можно дальше задачи решать с помощью их связи?


	- Knowing anatomy-connectivity relationships, we can make formal brain connection/functional alteration interpretations from anatomical 		associations with biological variables (of which there is a lot more than connectivity associations published). This could be done both 	as a review of existing literature, and as tool for new (e.g. the big ENIGMA) studies. Kind of my (BG’s) big goal with this project. 

### Данные:



Выборка HCP на которой просчитаны непрерывные коннектомы, на каждого человека 20к x 20к (~40) вершин,  всего ~1000 человек.

#### recon all:

-повоксельные лейблы роев (кортикальные+субкортикальные)


-регистрация в общее пространство (матрица 4x4 пространство MNI)


-геометрические модели кортикальных поверхностей белого и серого вещества


-толщина площадь и элемент объема для каждой точки сетки

  * csd - constrains spherical deconvolutions

