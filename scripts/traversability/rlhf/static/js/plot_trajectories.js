// plot_trajectories.js
document.addEventListener("DOMContentLoaded", function () {
    let trajectories = [];
    let labels = [];
    let bevImage = null;  // To store the BEV image in base64 format
    let seq, frame;  // Variables to store the sequence and frame numbers
    let sampleIndex = -1;  // Variable to store the current sample index

    // Function to plot trajectories using Plotly.js with the BEV image as background
    function plotTrajectoriesOnBEV() {
        const layout = {
            title: 'BEV Trajectories',
            xaxis: {
                range: [-128, 128],  // Center the plot by adjusting the range
                showgrid: false,
                zeroline: false,
                showticklabels: false  // Hide tick labels if not needed
            },
            yaxis: {
                range: [-128, 128],  // Invert the y-axis to center the image and trajectories
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                scaleanchor: "x",  // Keep the aspect ratio fixed
            },
            images: [{
                source: `data:image/png;base64,${bevImage}`,  // Add BEV image as the background
                xref: "x",
                yref: "y",
                x: -128,  // Move the image's center to (0, 0)
                y: 128,  // Adjust so the image is centered in the plot
                sizex: 256,  // Match the width of the plot
                sizey: 256,  // Match the height of the plot
                sizing: "stretch",  // Stretch to cover the plot area
                opacity: 1,  // Make it fully visible
                layer: "below"  // Make sure the image is behind the trajectories
            }],
            margin: { l: 0, r: 0, t: 40, b: 0 }  // Adjust margins if needed
        };

        let traces = [];
        trajectories.forEach((trajectory, index) => {
            let trace = {
                x: trajectory.map(point => point[1] - 128),  // Shift x coordinates to center
                y: trajectory.map(point => -(point[0] - 128)),  // Shift y coordinates to center
                mode: 'markers+lines+text',  // Add 'text' to show labels
                type: 'scatter',
                name: `${index}`,
                marker: {
                    symbol: labels[index] === 'optimal' ? 'circle' : 'cross',
                    size: 4,
                    color: labels[index] === 'optimal' ? 'green' : 'red'
                },
                line: {
                    color: labels[index] === 'optimal' ? 'green' : 'red'
                },
                // Add the trajectory number next to the last point of the trajectory
                text: trajectory.map((_, i) => i === trajectory.length - 5 ? `${index}` : ''),  // Label only the last point
                textposition: 'top',  // Position the text next to the line
                textfont: {
                    family: 'Arial, sans-serif',
                    size: 20,
                    color: 'white'
                }
            };
            traces.push(trace);
        });

        // Create the Plotly plot with the BEV image and trajectories
        Plotly.newPlot('bev-plot', traces, layout);
    }

    // Function to dynamically adjust the height of the list based on the number of items
    function adjustListHeight() {
        const listElement = document.getElementById('label-list');
        const listItems = document.querySelectorAll('#trajectory-list li');
        const listItemHeight = 40;  // Estimated height of each list item in pixels

        // Calculate the height dynamically based on the number of items
        const totalHeight = listItems.length * listItemHeight;

        // Set the new height, with a maximum height limit
        listElement.style.height = Math.min(totalHeight, 500) + 'px';  // Limit to 500px for overflow
    }

    // Example function to update the trajectory list dynamically
    function updateTrajectoryList() {
        const list = document.getElementById('trajectory-list');
        list.innerHTML = '';  // Clear existing list
        trajectories.forEach((_, index) => {
            const li = document.createElement('li');
            li.textContent = `Trajectory ${index} - ${labels[index]}`;
            li.dataset.index = index;  // Store the index

            // Set background color based on whether it's optimal or suboptimal
            if (labels[index] === 'optimal') {
                li.style.backgroundColor = 'green';  // Optimal -> Green background
                li.style.color = 'white';  // Ensure the text is visible
            } else {
                li.style.backgroundColor = 'red';  // Suboptimal -> Red background
                li.style.color = 'white';  // Ensure the text is visible
            }


            // Add event handlers for hovering and clicking (as before)
            li.onmouseover = function () {
                darkenTrajectoryColor(index);
            };
            li.onmouseout = function () {
                resetTrajectoryColor(index);
            };
            li.onclick = function () {
                toggleOptimalSuboptimal(index);
            };

            list.appendChild(li);
        });

        // After updating the list, adjust the height
        adjustListHeight();
    }

    // Function to darken the color of the trajectory on hover
    function darkenTrajectoryColor(index) {
        const darkColor = labels[index] === 'optimal' ? 'darkgreen' : 'darkred';
        Plotly.restyle('bev-plot', {
            'marker.color': darkColor,
            'line.color': darkColor
        }, [index]);
    }

    // Function to reset the color of the trajectory on mouseout
    function resetTrajectoryColor(index) {
        const originalColor = labels[index] === 'optimal' ? 'green' : 'red';
        Plotly.restyle('bev-plot', {
            'marker.color': originalColor,
            'line.color': originalColor
        }, [index]);
    }

    // Function to toggle between optimal/suboptimal for a trajectory
    function toggleOptimalSuboptimal(index) {
        // Toggle the label
        labels[index] = labels[index] === 'optimal' ? 'suboptimal' : 'optimal';

        // Update the marker symbol and color on the plot
        Plotly.restyle('bev-plot', {
            'marker.symbol': labels[index] === 'optimal' ? 'circle' : 'cross',
            'marker.color': labels[index] === 'optimal' ? 'green' : 'red',
            'line.color': labels[index] === 'optimal' ? 'green' : 'red'
        }, [index]);

        // Update the text in the list
        const listItem = document.querySelector(`li[data-index='${index}']`);
        listItem.textContent = `Trajectory ${index} - ${labels[index]}`;

        // Update the background color based on the new label
        if (labels[index] === 'optimal') {
            listItem.style.backgroundColor = 'green';
            listItem.style.color = 'white';  // Ensure text is visible
        } else {
            listItem.style.backgroundColor = 'red';
            listItem.style.color = 'white';  // Ensure text is visible
        }
    }

    // Function to load the next trajectory
    function loadNextTrajectory(index, regen = '0') {
        sampleIndex = index;
        // Add both args to the URL
        const url = `/load?index=${index}&regen=${regen}`;

        fetch(url)
            .then(response => response.json())
            .then(data => {
                // Update trajectories and labels
                trajectories = data.trajectories;
                // Map labels from ranking to optimal/suboptimal
                labels = data.labels.map(label => label === 0 ? 'optimal' : 'suboptimal');
                seq = String(data.seq);
                frame = String(data.frame);
                sampleIndex = data.index;

                // Ensure the BEV image is updated from the server's response
                bevImage = data.bev_image;  // This should now contain the base64-encoded image string

                // Update the front-view image
                document.getElementById('front-image').src = `data:image/png;base64,${data.front_image}`;

                // Call the function to plot the trajectories and the BEV image
                plotTrajectoriesOnBEV();  // Plot the new trajectories
                updateTrajectoryList();  // Update the trajectory list
                displaySeqFrame();  // Display the current sequence and frame
            });
    }

    // Function to display sequence and frame
    function displaySeqFrame() {
        const displayElement = document.getElementById('seq-frame-display');
        displayElement.textContent = `Sample Index: ${sampleIndex} Sequence: ${seq}, Frame: ${frame}`;
    }

    // Function to show the custom alert
    function showAlert(message) {
        const alertBox = document.getElementById('custom-alert');
        alertBox.textContent = message;  // Set the message
        alertBox.style.display = 'block';  // Show the alert
        // Hide the alert after 1 second (1000 ms)
        setTimeout(function () {
            alertBox.style.display = 'none';
        }, 1000);
    }

    // Handle "Go To Index" button click
    document.getElementById('go-to-index-btn').addEventListener('click', function () {
        index = document.getElementById('trajectory-index-input').value;
        if (index !== '') {
            loadNextTrajectory(index);
        }
    });

    // Fetch the initial trajectories when the "Generate Trajectories" button is clicked
    document.getElementById('generate-btn').addEventListener('click', function () {
        loadNextTrajectory(sampleIndex, regen = '1');
    });

    // Load the next trajectory when the "Next Trajectory" button is clicked
    document.getElementById('next-btn').addEventListener('click', function () {
        // nextSampleIndex = parseInt(sampleIndex) + 1;
        loadNextTrajectory(String(-1));
    });

    // Save the trajectories and labels
    document.getElementById('save-btn').addEventListener('click', function () {
        labels_int = labels.map(label => label === 'optimal' ? 0 : 1);
        fetch('/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ trajectories: trajectories, labels: labels_int })
        }).then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Show temporary alert for successful save
                    showAlert(`Seq ${data.seq}, Frame ${data.frame} saved successfully!`);
                }
            });
    });
});