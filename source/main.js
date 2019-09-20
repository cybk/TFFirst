function createModel () {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single hidden layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    //Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));
    return model;
}

async function getData () {
    const houseDataReq = await fetch('https://raw.githubusercontent.com/meetnandu05/ml1/master/house.json');
    const houseData = await houseDataReq.json();
    const cleaned = houseData
    .map((item) => ({
       price: item.Price,
       rooms: item.AvgAreaNumberofRooms 
    }))
    .filter(house => (house.price != null && house.rooms != null));

    return cleaned;
}

async function run () {
    // Load and plot the original input data
    const data = await getData();
    const values = data.map(item => ({
        x: item.rooms,
        y: item.price
    }));

    tfvis.render.scatterplot(
        { name: 'No. of Rooms vs Price'},
        { values },
        {
            xLabel: 'No. of Rooms',
            yLabel: 'Price',
            heignt: 500
        }
    );

    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);
}

document.addEventListener('DOMContentLoaded', run);