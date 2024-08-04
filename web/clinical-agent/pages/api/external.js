import React from 'react';
import { Grid, Card, CardContent, Typography } from '@mui/material';

// A function to display the third level items vertically inside a card
const DetailCard = ({ title, items }) => (
  <Card variant="outlined" sx={{ margin: 2 }}>
    <CardContent>
      <Typography gutterBottom variant="h6" component="div">
        {title}
      </Typography>
      {items.map((item, index) => (
        <Typography key={index} variant="body1">
          {item}
        </Typography>
      ))}
    </CardContent>
  </Card>
);

// The main component to export
export const ExternalSources = () => {
  const data = {
    Drug: ['Drugbank', 'PubChem', 'KEGG'],
    Disease: ['ICD-11', 'Disease Ontology'],
    KnowledgeGraph: ['Hetionet', 'DRKG'],
    Other: ['ClinicalTrials.gov', 'GPT Generated data'],
  };

  const model = ['Outcome binary prediction', 'Failure reason prediction', 'Time prediction', 'SMILES to ADMET'];

  return (
    <Grid container spacing={3}>
      {/* Data Section */}
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>Data</Typography>
        <Grid container>
          {Object.entries(data).map(([key, values], index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <DetailCard title={key} items={values} />
            </Grid>
          ))}
        </Grid>
      </Grid>

      {/* Model Section */}
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>Model</Typography>
        <Grid container>
          <Grid item xs={12}>
            <DetailCard items={model} />
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
};

export default ExternalSources;
