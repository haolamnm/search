package none.services;

import java.util.Map;

import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;

@Path("/")
public class Server {

    @GET
    @Path("/ping")
    @Produces(MediaType.APPLICATION_JSON)
    public Map<String, String> ping() {
        return Map.of("message", "pong");
    }
}
