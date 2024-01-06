import React, { useState } from 'react';
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
  import { useNavigate } from 'react-router-dom';
  
import axios from "axios";

function Login() {const [email, setEmail] = useState(''); 
  const [password, setPassword] = useState('');
  const navigate = useNavigate();
console.log(process.env.REACT_APP_BASEURL);
  const handleClick = async () => {
    try {
      const postData = {value1:email,value2:password}
      const response = await axios.post(process.env.REACT_APP_BASEURL+'/login', postData)
      if (response.data.success) {
        // Redirect to home page upon successful login
        navigate('/home');
      } else {
        // Handle unsuccessful login
        console.error('Login failed');
      }
    } catch (error) {
      // Handle errors
      console.error('Error:', error);
    }
  

  };
  

  return (
    <MDBContainer className='cardbody'>

      <MDBRow  >
        <MDBCol >

          <MDBCard >
            <MDBCardBody   >

              <h2 >Login</h2>
              <p >Please enter your login and password!</p>

              <MDBInput label='Email address' id='formControlLgMail' type='email' value={email} size='lg' onChange={(e) => setEmail(e.target.value)} // Update email state on change
              />
              <MDBInput label='Password'id='formControlLgPass'type='password'value={password}size='lg'onChange={(e) => setPassword(e.target.value)} // Update password state on change
              />

              <p><a href="#!">Forgot password?</a></p>
              <MDBBtn onClick={handleClick}>
                Login
              </MDBBtn>


              <div>
                <p>Don't have an account? <a href="#!" class="text-white-50 fw-bold">Sign Up</a></p>

              </div>
            </MDBCardBody>
          </MDBCard>

        </MDBCol>
      </MDBRow>

    </MDBContainer>
  );
}
export default Login;
