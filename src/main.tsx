import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import './filters.css'
import './geographic.css'
import './predictive.css'
import './predictive-phase2.css'
import App from './App'
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
