function initCarte(gCarte, projection){
    

        let path = d3.geoPath().projection(projection);

    d3.json("custom.geo.json").then(function(data) {
        // Ajoutez des éléments de chemin pour chaque pays
    gCarte.selectAll("path")
        .data(data.features)
        .enter().append("path")
        .attr("d", path)  // Appliquez le générateur de chemins aux données géographiques
        .style("stroke", "black")
        .style("stroke-width", 0.5)
        .style("fill", "lightgray");
     
});
}