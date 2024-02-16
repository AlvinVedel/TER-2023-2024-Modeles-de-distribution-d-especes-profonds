
var affiche = true;
var affiche2 = false;
//var model = 'random forest';
var g2;
var pa_guess;
var liste_plus_presentes;

function initPA(gTriangles, gTriangles2, model) {
    g2 = gTriangles2;
    d3.json("donnees_abio_especes.json").then(function(data) {
        for (var i = 0; i < data.length; i++) {
            var coords = projection([data[i].lon, data[i].lat]);
            let especes = "Espèces présentes : ";
            
            for (var j = 0; j < data[i].speciesId.length - 1; j++) {
                especes += data[i].speciesId[j]+", ";
            }
            especes += data[i].speciesId[data[i].speciesId.length-1];
            gTriangles.append("circle")
                .attr("cx", coords[0])
                .attr("cy", coords[1])
                .attr("class", "point")
                .attr("r", 1)  // Ajustez la taille du cercle selon vos besoins
                .attr("fill", "red")
                .on("click", function(d) {
                    d3.select(this).attr("stroke-width", "2px");  // Changer la couleur au clic                
                    // Afficher les informations dans le tooltip avec l'id "especes"
                    d3.select("#especes").html(especes);
                });
        }
        let dict = {};
        for(var i=0; i<data.length; i++){
            console.log("J'en suis au patchId n°"+i+" il y a "+data[i].speciesId.length+" espèces");
            for(var j=0; j<data[i].speciesId; j++){
                console.log("espèce n°")
                if(data[i].speciesId[j] in dict){
                    dict[data[i].speciesId[j]] += 1;
                }
                else{
                    dict[data[i].speciesId[j]] = 1; 
                }
            }
        }
        console.log(dict);
        var tempListe = [];
        for (var cle in dict) {
            tempListe.push({ cle: cle, valeur: dict[cle] });
        }

        tempListe.sort(function(a, b) {
            return b.valeur - a.valeur;
        });

        liste_plus_presentes = tempListe.map(function(item) {
            return item.cle;
        });
        console.log(liste_plus_presentes);
    });
    initPA_guess(model);
}


function initPA_guess(model){
    console.log(model);
    d3.json("donnees_abio_prediction.json").then(function(data) {
        afficherPA_toguess();
        for (var i = 0; i < data.length; i++) {
            var coords = projection([data[i].lon, data[i].lat]);
            let especes = listePredictions(data[i], model);
            pa_guess = g2.append("circle")
                .attr("cx", coords[0])
                .attr("cy", coords[1])
                .attr("class", "point_guess")
                .attr("r", 1)  // Ajustez la taille du cercle selon vos besoins
                .attr("fill", "blue")
                .on("click", function(d) {
                    d3.select(this).attr("stroke-width", "2px");
                    d3.select(this).attr("fill", "lightblue");  // Changer la couleur au clic                
                    // Afficher les informations dans le tooltip avec l'id "especes"
                    d3.select("#especes").html(especes);
                })
                .style("display", "none");
        }
        
        afficherPA_toguess();
    });
}

function afficherPA(){
    affiche = !affiche;
    if(affiche){
        d3.selectAll(".point")
            .style("display", "block");
    }
    else{
        d3.selectAll(".point")
            .style("display", "none");
    } 
}

function afficherPA_toguess(){
    affiche2 = !affiche2;
    if(affiche2){
        d3.selectAll(".point_guess")
            .style("display", "block");
    }
    else{
        d3.selectAll(".point_guess")
            .style("display", "none");
    } 
}

function listePredictions(patchID, model){
    let especes = "Espèces prédites avec "+model+" : ";
    if(model=="random forest"){
        for (var j = 0; j < patchID.Prediction_random_forest.length - 1; j++) {
        especes += patchID.Prediction_random_forest[j]+", ";
        }
    }
    else if(model=="knn"){
        for (var j = 0; j < patchID.Prediction_knn.length - 1; j++) {
            especes += patchID.Prediction_knn[j]+", ";
            }
    }
    else if(model=="knn abiotique"){
        for (var j = 0; j < patchID.Prediction_knn_abiotique.length - 1; j++) {
            especes += patchID.Prediction_knn_abiotique[j]+", ";
            }
    }
    else if(model=="svm"){
        for (var j = 0; j < patchID.Prediction_SVM.length - 1; j++) {
            especes += patchID.Prediction_SVM[j]+", ";
            }
    }
    else{
        especes = "Veuillez sélectionner un modèle";
    }
    return especes;
}

function changerModel(string){
    if(string=='knn'){
        model = 'knn';
    }
    else if(string=="knn_abiotique"){
        model='knn abiotique';
    }
    else if(string=='svm'){
        model='svm';
    }
    else if(string=='random_forest'){
        model='random forest';
    }
    else{
        model = 'invalid model';
    }
    initPA_guess(model);
    
}