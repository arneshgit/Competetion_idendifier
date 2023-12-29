import React, { useState } from 'react';
import './App.css';
import {
  MDBBtn,
  MDBContainer,
  MDBRow,
  MDBCol,
  MDBCard,
  MDBCardBody,
  MDBInput,
  MDBIcon
}
  from 'mdb-react-ui-kit';
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from "./pages/login";
import Home from "./pages/home";

function App() {
  return(
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} /> {/* Route to the Login component */}
        <Route path="/home" element={<Home />} />
      </Routes>
    </Router>
  );
  
}

export default App;